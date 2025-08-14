# models/ner_processor.py
import json
import logging
import re
from typing import List, Dict, Union
from huggingface_hub import hf_hub_download
import os
import torch
import torch.nn as nn
from torchcrf import CRF
from transformers import (
    AutoTokenizer, AutoConfig,
    XLMRobertaPreTrainedModel, XLMRobertaModel
)
import string

logger = logging.getLogger(__name__)
TR_PUNCT = set(string.punctuation) | {"’","“","”","…","–","—","«","»","‹","›"}


APOS = {"’": "'", "‘": "'", "ʼ": "'"}

def _norm_apostrophes(s: str) -> str:
    for b,g in APOS.items():
        s = s.replace(b,g)
    return s

def _spacey_apostrophe_variant(s: str) -> tuple[str, list[int]]:
    """
    "İstanbul'da" -> "İstanbul 'da"
    Ayrıca t2o (transformed->original) char hizalama haritası döner.
    Eklenen boşlukların index'i için t2o'da önceki orijinal index'i kopyalarız.
    """
    s = _norm_apostrophes(s)
    out = []
    t2o = []
    i = 0
    while i < len(s):
        ch = s[i]
        out.append(ch); t2o.append(i)
        if ch.isalpha() or ch.isdigit():
            # bir sonraki char apostrof mu?
            if i+1 < len(s) and s[i+1] == "'":
                # BOŞLUK EKLE
                out.append(" ")
                t2o.append(i)  # yeni boşluğu mevcut harfe hizala (yakın komşu)
        i += 1
    return "".join(out), t2o

def _identity_variant(s: str) -> tuple[str, list[int]]:
    s2 = _norm_apostrophes(s)
    return s2, list(range(len(s2)))


# ----- CRF'li model sınıfı (train'de kullandığının birebir aynısı) -----
class XLMRForTokenClassificationCRF(XLMRobertaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.roberta = XLMRobertaModel(config, add_pooling_layer=False)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, self.num_labels)
        self.crf = CRF(self.num_labels, batch_first=True)
        self.post_init()

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        out = self.roberta(input_ids, attention_mask=attention_mask, **kwargs)
        seq = self.dropout(out.last_hidden_state)
        emissions = self.classifier(seq)
        mask = attention_mask.bool() if attention_mask is not None else None
        if labels is not None:
            loss = -self.crf(emissions, labels, mask=mask, reduction="mean")
            preds = self.crf.decode(emissions, mask=mask)
            return {"loss": loss, "logits": emissions, "predictions": preds}
        else:
            preds = self.crf.decode(emissions, mask=mask)
            return {"logits": emissions, "predictions": preds}


class NERProcessor:
    """
    CRF başlıklı, lokalde ince ayarlı Türkçe NER modeli ile çıkarım.
    PERSON/ORGANIZATION/LOCATION + MONEY/DATE_TIME/LEGAL_REF/PHONE_EMAIL destekler.
    """

    def __init__(self, model_dir: str = "models/model_ner"):
        self.model_dir = model_dir
        self.model = None
        self.tokenizer = None
        self.device = self._get_device()
        self.model_loaded = False
        self.space_var = False

        # (UI için)
        self.entity_types = {
            'PERSON': {'name': 'Kişiler', 'icon': 'fas fa-user'},
            'ORGANIZATION': {'name': 'Organizasyonlar', 'icon': 'fas fa-building'},
            'LOCATION': {'name': 'Lokasyonlar', 'icon': 'fas fa-map-marker-alt'},
            'MONEY': {'name': 'Para Birimleri', 'icon': 'fas fa-money-bill'},
            'DATE_TIME': {'name': 'Tarihler', 'icon': 'fas fa-calendar'},
            'LEGAL_REF': {'name': 'Hukuki Atıf', 'icon': 'fas fa-scale-balanced'},
            'PHONE_EMAIL': {'name': 'İletişim', 'icon': 'fas fa-envelope'}
        }

        # Apostrof son ekleri
        self.SUFFIXES = {
            "'da","'de","'ta","'te","'ya","'ye",
            "'nın","'nin","'nun","'nün",
            "'dır","'dir","'dur","'dür",
            "'dan","'den","'tan","'ten"
        }
        # Ek genişletmeyi uygulayacağımız türler
        self._SUFFIX_TYPES = {"LOCATION", "ORGANIZATION", "PERSON", "LEGAL_REF","PHONE_EMAIL","DATE_TIME","MONEY"}

        # Hukuk kısaltmaları 
        self.LAW_ACRONYMS = {"TTK","TCK","TMK","CMK","İİK","İIK","HMK","YYK","KVKK","VUK","SGK"}

    def _map_span_back(self, start: int, end: int, t2o: list[int], orig_len: int) -> tuple[int,int]:
        # start/end'i transformed->original indexlerine çevirirken,
        # araya serpiştirilen boşluklara denk gelirse, en yakın geçerli index'e yasla.
        def map_one(pos):
            pos = max(0, min(pos, len(t2o)-1))
            o = t2o[pos]
            # güvenlik: None yok ama yine de sınırla
            return max(0, min(o, orig_len))
        return map_one(start), map_one(end-1)+1  # end exclusive

    def _infer_once(self, text: str) -> list[dict]:
        # CRF’li mevcut inference akışındaki core kısmını kullanan tek seferlik tahmin.
        enc = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=320,
                            return_offsets_mapping=True)
        offsets = enc.pop("offset_mapping")[0].tolist()
        # Device'ı güvenli şekilde al
        try:
            dev = next(self.model.parameters()).device
        except StopIteration:
            dev = torch.device(self.device)
        enc = {k: v.to(dev) for k,v in enc.items()}
        with torch.no_grad():
            out = self.model(**enc)
            pred_ids = out["predictions"][0]
        return self._decode_bio_from_subtokens(text, pred_ids, offsets)  # senin mevcut fonksiyonun


    def _get_device(self):
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"

    def load_model(self) -> bool:
        """Lokal veya Hugging Face'ten CRF’li modeli ve etiketleri yükle."""
        try:
            logger.info(f"Loading NER model from: {self.model_dir}")

            # Etiket haritaları
            if os.path.exists(f"{self.model_dir}/extended_label_mappings.json"):
                with open(f"{self.model_dir}/extended_label_mappings.json", "r") as f:
                    maps = json.load(f)
            else:
                # HF repo'dan çek
                file_path = hf_hub_download(repo_id=self.model_dir, filename="extended_label_mappings.json")
                with open(file_path, "r") as f:
                    maps = json.load(f)

            id2label = {int(k): v for k, v in maps["id2label"].items()}
            label2id = maps["label2id"]

            # Tokenizer & Config (local dosyalardan yükle)
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir, local_files_only=True)
            config = AutoConfig.from_pretrained(
                self.model_dir, local_files_only=True,
                id2label=id2label, label2id=label2id
            )

            self.model = XLMRForTokenClassificationCRF.from_pretrained(
                self.model_dir, config=config, local_files_only=True
            )
            dev = torch.device(self.device)
            self.model.to(dev).eval()

            self.id2label = id2label
            self.label2id = label2id
            self.model_loaded = True
            logger.info("NER model loaded successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to load NER model: {e}", exc_info=True)
            self.model_loaded = False
            return False
    def _trim_all_whitespace_punct(self, entities, text):
        out = []
        n = len(text)
        for e in entities:
            s, ed = max(0, e['start']), min(n, e['end'])
            # solda boşluk/noktalama kırp
            while s < ed and (text[s].isspace() or text[s] in TR_PUNCT):
                s += 1
            # sağda boşluk/noktalama kırp
            while ed > s and (text[ed-1].isspace() or text[ed-1] in TR_PUNCT):
                ed -= 1
            if ed > s:
                e = {**e, 'start': s, 'end': ed, 'text': text[s:ed]}
                out.append(e)
        return out

    def _trim_apostrophe_suffixes(self, entities, text):
        SUF = {
            "da","de","ta","te","ya","ye",
            "nın","nin","nun","nün",
            "dır","dir","dur","dür",
            "dan","den","tan","ten"
        }
        TYPES = {"LOCATION","ORGANIZATION","PERSON","LEGAL_REF"}  # istersen daraltabilirsin
        out = []
        for ent in entities:
            if ent["type"] not in TYPES:
                out.append(ent); continue
            s, e = ent["start"], ent["end"]
            frag = text[s:e]
            # apostrof + ek varsa kes
            m = re.search(r"^(.*)\'([A-Za-zçğıöşüÇĞİÖŞÜ]+)$", frag)
            if m:
                stem, suf = m.group(1), m.group(2).lower()
                if suf in SUF:
                    new_end = s + len(stem)  # apostrof öncesine kadar
                    if new_end > s:
                        ent = {**ent, "end": new_end, "text": text[s:new_end]}
            out.append(ent)
        return out



    def analyze_entities(self, text: str) -> Dict[str, Union[bool, List[Dict], str]]:
        if not self.model_loaded and not self.load_model():
            return {'success': False, 'error': 'NER model not available', 'entities': [], 'entity_count': 0}

        try:
            if not text or not text.strip():
                return {'success': True, 'entities': [], 'entity_count': 0, 'error': None}

            orig = text

            # --- Varyant A: boşluksuz (senin eğitim formatın) ---
            a_text, a_t2o = _identity_variant(orig)
            a_ents = self._infer_once(a_text)
            # a_ents zaten a_text üzerinde; orijinale hizala:
            a_ents_mapped = []
            for e in a_ents:
                s2, e2 = self._map_span_back(e['start'], e['end'], a_t2o, len(orig))
                if e2 > s2:
                    a_ents_mapped.append({**e, 'start': s2, 'end': e2, 'text': orig[s2:e2]})

            entities = a_ents_mapped
            # --- Varyant B: Stefan’a yakın (apostroftan önce boşluk) ---
            if self.space_var:
                b_text, b_t2o = _spacey_apostrophe_variant(orig)
                b_ents = self._infer_once(b_text)
                b_ents_mapped = []
                for e in b_ents:
                    s2, e2 = self._map_span_back(e['start'], e['end'], b_t2o, len(orig))
                    if e2 > s2:
                        b_ents_mapped.append({**e, 'start': s2, 'end': e2, 'text': orig[s2:e2]})

            # --- Birleştir + post-process ---
            

            # Apostrof ek genişletme, legal merge vs. (senin mevcut akışın)
            entities = self._trim_all_whitespace_punct(entities, orig)
            entities = self._expand_apostrophe_suffixes(entities, orig)
            entities = self._merge_legal_refs(entities, orig)
            entities = self._promote_legal_acronyms_for_law_context(entities, orig)

            # Pattern bazlı ekler
            entities.extend(self._extract_pattern_entities(orig))

            # Çakışma çözümü / tekilleştirme
            entities = self._deduplicate_entities(entities)
            entities.sort(key=lambda x: x.get('start', 0), reverse=False)

            return {'success': True, 'entities': entities, 'entity_count': len(entities), 'error': None}

        except Exception as e:
            logger.error(f"NER analysis failed: {e}", exc_info=True)
            return {'success': False, 'error': str(e), 'entities': [], 'entity_count': 0}


    # ---------------- helpers ----------------
    def _preprocess_text(self, text: str) -> str:
        text = re.sub(r'\s+', ' ', text)
        # eğik apostrofları düzleştir
        text = text.replace("’", "'").replace("‘", "'").replace("ʼ", "'")
        # ÖNEMLİ: apostrofu silme -> char class içine ' ekledik
        text = re.sub(r'[^\w\s\.,!?;:()\-\"\'@+%_]+', '', text)
        return text.strip()


    
    def _decode_bio_from_subtokens(self, text: str, pred_ids: List[int], offsets: List[List[int]]) -> List[Dict]:
        """CRF çıktı etiketlerini (BIO) subtoken offset’lerinden char span’lara çevirir."""
        ents: List[Dict] = []
        cur_type = None
        cur_start = None

        for i, lab_id in enumerate(pred_ids):
            start, end = offsets[i]
            if end <= start:  # özel token veya boş
                continue
            label = self.id2label.get(lab_id, "O")

            if label == "O":
                if cur_type is not None:
                    ents.append(self._make_ent(text, cur_type, cur_start, prev_end=end))
                    cur_type, cur_start = None, None
                continue

            # label -> (B-FOO / I-FOO)
            if "-" in label:
                bi, typ = label.split("-", 1)
            else:
                bi, typ = "B", label

            if bi == "B" or (cur_type is not None and typ != cur_type):
                if cur_type is not None:
                    ents.append(self._make_ent(text, cur_type, cur_start, prev_end=offsets[i-1][1]))
                cur_type = typ
                cur_start = start
            else:
                # I- tipi devam; sadece genişlet
                pass

        if cur_type is not None and cur_start is not None:
            ents.append(self._make_ent(text, cur_type, cur_start, prev_end=offsets[len(pred_ids)-1][1]))

        return ents

    def _make_ent(self, full_text: str, typ: str, s: int, prev_end: int) -> Dict:
        s = max(0, s)
        e = max(s, prev_end)
        frag = full_text[s:e]
        return {
            'text': frag,
            'type': typ,
            'start': s,
            'end': e,
            'source': 'model'
        }

    # ---------- POST-PROCESS: Apostrof ek genişletme ----------
    def _expand_apostrophe_suffixes(self, entities: List[Dict], text: str) -> List[Dict]:
        out = []
        for ent in entities:
            if ent["type"] not in self._SUFFIX_TYPES:
                out.append(ent); continue
            e = ent["end"]
            # En uzun eki önce dene
            for suf in sorted(self.SUFFIXES, key=len, reverse=True):
                if text[e:e+len(suf)] == suf:
                    e += len(suf)
                    break
            if e != ent["end"]:
                ent = {**ent, "end": e, "text": text[ent["start"]:e]}
            out.append(ent)
        return out

    # ---------- POST-PROCESS: LEGAL_REF birleştirme ----------
    def _merge_legal_refs(self, entities: List[Dict], text: str) -> List[Dict]:
        if not entities: return entities
        merged = []
        i = 0
        while i < len(entities):
            cur = entities[i]
            if cur["type"] == "LEGAL_REF":
                j = i + 1
                end = cur["end"]
                # yakındaki ardışık LEGAL_REF'leri tek span'a çek
                while j < len(entities) and entities[j]["type"] == "LEGAL_REF" and entities[j]["start"] <= end + 2:
                    end = max(end, entities[j]["end"])
                    j += 1
                if j > i + 1:
                    merged.append({
                        "text": text[cur["start"]:end],
                        "type": "LEGAL_REF",
                        "confidence": min(cur["confidence"], entities[j-1]["confidence"]),
                        "start": cur["start"],
                        "end": end,
                        "source": "model"
                    })
                    i = j
                    continue
            merged.append(cur)
            i += 1
        return merged

    # ---------- POST-PROCESS: Hukuk kısaltmalarını LEGAL_REF'e terfi ----------
    def _promote_legal_acronyms_for_law_context(self, entities: List[Dict], text: str) -> List[Dict]:
        if not entities: return entities
        out = []
        for idx, ent in enumerate(entities):
            if ent["type"] in {"ORGANIZATION","MISC"}:
                token = ent["text"].strip().upper()
                if token in self.LAW_ACRONYMS:
                    # Çevre bağlamına bak: yakınında "sayılı" veya "madde" geçiyor mu?
                    left = max(0, ent["start"] - 20)
                    right = min(len(text), ent["end"] + 20)
                    ctx = text[left:right].lower()
                    if ("sayılı" in ctx) or ("madde" in ctx) or ("maddesi" in ctx) or ("md." in ctx):
                        ent = {**ent, "type": "LEGAL_REF"}
            out.append(ent)
        return out

    # ---------- Pattern tabanlı yakalamalar ----------
    def _extract_pattern_entities(self, text: str) -> List[Dict]:
        entities: List[Dict] = []

        # Tarih
        date_patterns = [
            r'\b\d{1,2}[./]\d{1,2}[./]\d{4}\b',
            r'\b\d{4}[./]\d{1,2}[./]\d{1,2}\b',
            r'\b\d{1,2}\s+(Ocak|Şubat|Mart|Nisan|Mayıs|Haziran|Temmuz|Ağustos|Eylül|Ekim|Kasım|Aralık)\s+\d{4}\b'
        ]
        for p in date_patterns:
            for m in re.finditer(p, text, re.IGNORECASE):
                entities.append({'text': m.group(), 'type': 'DATE_TIME',
                                 'start': m.start(), 'end': m.end(), 'source': 'pattern'})

        # Para
        money_patterns = [
            r'\b\d{1,3}(?:[.\s]\d{3})(?:[.,]\d+)?\s(?:TL|lira|EUR|USD|₺|\$|€)\b',
            r'\b(?:TL|lira|EUR|USD|₺|\$|€)\s*\d+(?:[.,]\d+)?\b'
        ]
        for p in money_patterns:
            for m in re.finditer(p, text, re.IGNORECASE):
                entities.append({'text': m.group(), 'type': 'MONEY',
                                 'start': m.start(), 'end': m.end(), 'source': 'pattern'})

        # E-posta (model kaçırsa bile @ varsa ekle)
        if '@' in text:
            email_pat = re.compile(
                r'''(?<![\w\.\-\+])
                    [A-Za-z0-9._%+\-]+
                    @
                    [A-Za-z0-9.\-]+
                    \.[A-Za-z]{2,}(?:\.[A-Za-z]{2,})*
                ''', re.VERBOSE
            )
            for m in email_pat.finditer(text):
                entities.append({'text': m.group(), 'type': 'PHONE_EMAIL',
                                 'start': m.start(), 'end': m.end(), 'source': 'pattern'})

        # Telefon
        phone_pat = r'\b0\d{3}\s?\d{3}\s?\d{2}\s?\d{2}\b'
        for m in re.finditer(phone_pat, text):
            entities.append({'text': m.group(), 'type': 'PHONE_EMAIL',
                             'start': m.start(), 'end': m.end(), 'source': 'pattern'})

        return entities

    def _deduplicate_entities(self, entities: List[Dict]) -> List[Dict]:
        if not entities: return entities
        # Başlangıca göre sırala; daha sonra çakışanlarda en uzun ve yüksek confident’i seç
        entities.sort(key=lambda x: (x['start'], -(x['end']-x['start'])))
        out: List[Dict] = []
        for ent in entities:
            if not ent.get("text"): 
                continue
            drop = False
            for i, ex in enumerate(out):
                overlap = not (ent['end'] <= ex['start'] or ex['end'] <= ent['start'])
                same_text = ent['text'].strip().lower() == ex['text'].strip().lower()
                if overlap or same_text:
                    # öncelik: daha uzun span
                    if (ent['end']-ent['start'] > ex['end']-ex['start']):
                        out[i] = ent
                    drop = True
                    break
            if not drop:
                out.append(ent)
        return out

    def get_entity_summary(self, entities: List[Dict]) -> Dict[str, int]:
        summary = {}
        for e in entities:
            summary[e['type']] = summary.get(e['type'], 0) + 1
        return summary

    def is_model_ready(self) -> bool:
        return self.model_loaded


# Global instance
ner_processor = NERProcessor()