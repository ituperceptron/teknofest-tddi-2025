# GovAI - Türkçe AI Metin İşleme Platformu

GovAI, **Teknofest yarışması** için geliştirilmiş kapsamlı bir **offline AI metin işleme platformu**dur. Flask tabanlı bu web uygulaması, gelişmiş AI modellerini kullanarak Türkçe metinler üzerinde çeşitli işlemler yapabilmektedir.

##  Ana Özellikler

###  AI Destekli İşlemler
-  **Metin Özetleme**: LLaMA 3.1 8B modeliyle gelişmiş Türkçe özetleme
-  **Metin Sınıflandırma**: XLM-RoBERTa ile resmi evrak türlerini tanıma
-  **OCR (Optik Karakter Tanıma)**: Qwen2.5-VL ile PDF ve resimlerden metin çıkarma
-  **NER (Varlık Tanıma)**: Turkish BERT ile kişi, yer, kurum tanıma

###  Platform Özellikleri
-  **Kullanıcı Kimlik Doğrulama**: Güvenli kayıt ve giriş sistemi
-  **Dosya Yönetimi**: PDF ve resim dosyası yükleme/işleme
-  **İşlem Geçmişi**: Tüm AI işlemlerinin detaylı takibi
-  **Responsive Tasarım**: Mobil ve masaüstü uyumlu modern arayüz
-  **Offline Çalışma**: İnternet bağlantısı gerektirmez
-  **Çoklu Dosya İşleme**: Toplu PDF işleme desteği

##  Teknoloji Stack

### Backend
- **Framework**: Flask (Python)
- **Veritabanı**: SQLite (offline)
- **AI Framework**: PyTorch, Transformers, llama-cpp-python
- **PDF İşleme**: PyMuPDF (fitz)
- **Görüntü İşleme**: Pillow
- **Diğer**: Accelerate, Hugging Face Hub, Safetensors

### AI Modelleri
- **Özetleme**: Meta LLaMA 3.1 8B Instruct (GGUF)
- **Sınıflandırma**: joeddav/xlm-roberta-large-xnli
- **OCR**: Qwen2.5-VL-3B-Instruct (yerel model)
- **NER**: [ituperceptron/turkish-ner-itu-perceptron](https://huggingface.co/ituperceptron/turkish-ner-itu-perceptron)
- **NER (No CRF Version)** [ituperceptron/turkish-ner-itu-perceptron-no-crf](https://huggingface.co/ituperceptron/turkish-ner-itu-perceptron-no-crf)
- **NER Dataset**: [ituperceptron/turkish-ner-dataset](https://huggingface.co/ituperceptron/turkish-ner-itu-perceptron)
### Frontend
- **UI**: HTML5, CSS3, JavaScript (ES6+)
- **İkonlar**: Font Awesome 6
- **Tasarım**: Mobile-first responsive design

##  Kurulum

### Sistem Gereksinimleri

- **Python**: 3.8 veya üzeri
- **RAM**: Minimum 8GB (16GB+ önerilen)
- **Depolama**: ~15GB (AI modelleri için)
- **GPU**: CUDA destekli GPU (opsiyonel, performans için). Apple Silicon (M1/M2/M3) üzerinde Metal/MPS hızlandırma desteklenir.

### Kurulum Adımları

#### 1. Projeyi Klonlayın
```bash
git clone <repository-url>
cd tkfest_y-2
```

#### 2. Python Sanal Ortamı Oluşturun
```bash
# Sanal ortam oluşturma
python -m venv venv

# Aktivasyon
# Windows:
venv\Scripts\activate

# macOS/Linux:
source venv/bin/activate
```

#### 3. Bağımlılıkları Yükleyin

PyTorch kurulumunu işletim sisteminize göre resmi yönergelerle yapmanız önerilir. Ardından diğer paketleri yükleyin.

```bash
# (Önerilir) Önce PyTorch'u kurun
# CUDA'lı Linux/Windows için: https://pytorch.org/get-started/locally/
# macOS (Apple Silicon) için genellikle CPU/MPS kurulum yeterlidir

# Ardından proje bağımlılıkları
pip install -r requirements.txt
```

Gerekli başlıca paketler: `transformers (>=4.41,<5)`, `accelerate`, `huggingface-hub`, `safetensors`, `llama-cpp-python`, `PyMuPDF`, `Pillow`.

#### 4. AI Modellerini İndirin

**LLaMA 3.1 8B Model (GGUF, yerel):**
```bash
# models/ klasörü oluşturun
mkdir models

# Hugging Face'den LLaMA 3.1 8B GGUF modelini indirin
# https://huggingface.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF
# Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf dosyasını models/ klasörüne yerleştirin
```

**Qwen2.5-VL Model (yerel):**
```bash
# models/qwen_vlm klasörü oluşturun
mkdir -p models/qwen_vlm

# Qwen2.5-VL-3B-Instruct modelini indirin
# `models/qwen_vlm/` klasörüne tam model dosyalarını yerleştirin
# Not: Uygulama bu modeli internet olmadan sadece yerelden yükler (local_files_only=True)
```

#### 5. Uygulamayı Başlatın
```bash
python app.py
```

 **Tarayıcınızda açın:** http://localhost:5001

###  Hızlı Başlangıç

1. **Kayıt Olun**: İlk kez kullanıyorsanız "Kayıt Ol" ile hesap oluşturun
2. **Giriş Yapın**: Kullanıcı adı ve şifrenizle giriş yapın
3. **Model Yükleme**: İlk başlatmada AI modelleri yüklenecek (~2-5 dakika)
4. **Kullanmaya Başlayın**: Dashboard'dan istediğiniz AI özelliğini seçin

##  Kullanım Kılavuzu

###  AI Özellikleri

####  Metin Özetleme
- **Metin Girişi**: Doğrudan metin yazın veya PDF yükleyin
- **Çoklu PDF**: Birden fazla PDF'i aynı anda özetleyin
- **LLaMA 3.1**: Gelişmiş Türkçe özetleme için optimize edilmiş
- **Sonuç**: Özet metnini kopyalayın, kaydedin veya indirin

####  Metin Sınıflandırma
- **Evrak Türleri**: Şikayet dilekçesi, bilgi edinme başvurusu, sosyal yardım talebi vb.
- **PDF Desteği**: PDF dosyalarını otomatik sınıflandırma
- **Güven Skoru**: Her kategori için güvenilirlik oranı
- **Toplu İşlem**: Çoklu dosya sınıflandırması

####  OCR (Optik Karakter Tanıma)
- **PDF OCR**: Taranmış PDF'lerden metin çıkarma
- **Resim OCR**: JPG, PNG, BMP, TIFF formatlarını destekler
- **Qwen2.5-VL**: Gelişmiş vision model kullanımı
- **Türkçe Optimizasyon**: Türkçe karakterler için optimize edilmiş

####  NER (Varlık Tanıma)
- **Kişi Adları**: Metin içindeki kişi isimlerini tespit
- **Organizasyonlar**: Şirket, kurum, dernek adları
- **Lokasyonlar**: Şehir, ülke, adres bilgileri
- **Tarihler**: Çeşitli tarih formatlarını tanıma
- **Para Birimleri**: TL, USD, EUR vb. para ifadeleri
- **Hukuki Terimler**: Hukuki referanslar
- **İletişim bilgileri** Mail, telefon gibi iletişim bilgileri

###  Platform Kullanımı

#### Dashboard
- **Hızlı Erişim**: Tüm AI özelliklerine tek tıkla ulaşım
- **Son İşlemler**: En son 3 işleminizi görüntüleme
- **İstatistikler**: Toplam dosya ve işlem sayıları

#### Dosya Yönetimi
- **Yükleme**: Sürükle-bırak veya dosya seçici ile
- **İşlem Geçmişi**: Tüm dosyalarınızı ve sonuçlarını görüntüleme
- **Silme**: İstenmeyen dosyaları kaldırma
- **İndirme**: Sonuçları yerel olarak kaydetme

#### Profil
- **Kullanıcı Bilgileri**: Hesap detayları ve istatistikler
- **İşlem Sayıları**: Her AI özelliği için kullanım istatistikleri
- **Güvenlik**: Şifre değiştirme (gelecek sürüm)

##  Veritabanı Yapısı

### Ana Tablolar

#### `users` - Kullanıcı Bilgileri
- `id`: Birincil anahtar
- `username`: Kullanıcı adı (benzersiz)
- `password`: Şifre (hash'lenmiş)
- `email`: E-posta adresi
- `created_at`: Hesap oluşturma tarihi

#### `documents` - Genel Belgeler
- `id`: Birincil anahtar
- `user_id`: Kullanıcı ID'si (foreign key)
- `title`: Belge başlığı
- `content`: Belge içeriği
- `category`: Belge kategorisi
- `created_at`: Oluşturma tarihi

### AI İşlem Tabloları

#### `pdf_files` - Özetleme PDF'leri
- `id`, `user_id`, `original_filename`, `file_path`, `file_size`, `upload_date`

#### `classification_pdfs` - Sınıflandırma PDF'leri
- `id`, `user_id`, `original_filename`, `file_path`, `file_size`, `upload_date`

#### `ocr_pdfs` - OCR PDF'leri
- `id`, `user_id`, `original_filename`, `file_path`, `file_size`, `upload_date`

#### `ocr_images` - OCR Resimleri
- `id`, `user_id`, `original_filename`, `file_path`, `file_size`, `upload_date`

#### `ner_pdfs` - NER PDF'leri
- `id`, `user_id`, `original_filename`, `file_path`, `file_size`, `upload_date`

## 📁 Proje Yapısı

```
tkfest_y-2/
├── app.py                    #  Ana Flask uygulaması
├── database.db              #  SQLite veritabanı (otomatik oluşturulur)
├── requirements.txt         #  Python bağımlılıkları
├── README.md               #  Bu dosya
├── models/                 #  AI Model dosyaları
│   ├── __init__.py
│   ├── model_manager.py    #  Model yönetim sistemi
│   ├── classifier.py       #  Sınıflandırma modülü
│   ├── summarizer.py       #  Özetleme modülü
│   ├── ocr_processor.py    #  OCR modülü
│   ├── ner_processor.py    #  NER modülü
│   ├── Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf  # LLaMA model dosyası
│   └── qwen_vlm/           # Qwen2.5-VL model dosyaları
├── static/                 #  Statik dosyalar
│   ├── css/
│   │   └── style.css       # Ana CSS dosyası
│   ├── js/
│   │   └── script.js       # JavaScript işlevleri
│   └── images/
│       └── Teknofest_logo_pfp.png
├── templates/              #  HTML şablonları
│   ├── base.html           # Temel şablon
│   ├── index.html          # Ana sayfa
│   ├── login.html          # Giriş sayfası
│   ├── register.html       # Kayıt sayfası
│   ├── dashboard.html      # Kontrol paneli
│   ├── profile.html        # Profil sayfası
│   ├── summary.html        # Özetleme sayfası
│   ├── classification.html # Sınıflandırma sayfası
│   ├── ocr.html           # OCR sayfası
│   ├── ner.html           # NER sayfası
│   ├── previous_works.html # İşlem geçmişi
│   ├── documents.html     # Belgeler sayfası
│   └── my_classification_pdfs.html
└── uploads/               # 📤 Yüklenen dosyalar
    ├── summary_pdfs/      # Özetleme PDF'leri
    ├── classification_pdfs/ # Sınıflandırma PDF'leri
    ├── ocr_pdfs/         # OCR PDF'leri
    ├── ocr_images/       # OCR resimleri
    └── ner_pdfs/         # NER PDF'leri
```

## Özelleştirme

### Renk Teması
Ana renk temasını değiştirmek için `static/css/style.css` dosyasındaki CSS değişkenlerini düzenleyin:

```css
:root {
    --primary-color: #667eea;
    --secondary-color: #764ba2;
    --background-color: #f5f5f5;
    --text-color: #333;
}
```

### Veritabanı
SQLite veritabanı yerine başka bir veritabanı kullanmak için `app.py` dosyasındaki veritabanı bağlantısını değiştirin.

### Yeni Sayfa Ekleme
1. `templates/` klasörüne yeni HTML dosyası ekleyin
2. `app.py` dosyasına yeni route ekleyin
3. `base.html` dosyasındaki navigasyon menüsüne link ekleyin

## Güvenlik

- Şifreler hash'lenerek saklanmalıdır (production'da bcrypt kullanın)
- Session güvenliği için güçlü secret key kullanın
- SQL injection koruması için parametreli sorgular kullanılmıştır
- CSRF koruması eklenmelidir (production'da)

## Production Deployment

### Gunicorn ile Deployment
```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:8000 app:app
```

### Nginx ile Reverse Proxy
```nginx
server {
    listen 80;
    server_name your-domain.com;
    
    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

## Katkıda Bulunma

1. Fork yapın
2. Feature branch oluşturun (`git checkout -b feature/amazing-feature`)
3. Değişikliklerinizi commit edin (`git commit -m 'Add amazing feature'`)
4. Branch'inizi push edin (`git push origin feature/amazing-feature`)
5. Pull Request oluşturun

## Lisans

Bu proje MIT lisansı altında lisanslanmıştır. Detaylar için `LICENSE` dosyasına bakın.

## İletişim

Proje hakkında sorularınız için:
- GitHub Issues: [[Proje Issues Sayfası](https://github.com/ituperceptron/teknofest-tddi-2025/issues)]

## Changelog

### v1.0.0 (2024-01-01)
- İlk sürüm
- Temel kullanıcı kimlik doğrulama
- Metin özetleme ve sınıflandırma
- Belge yönetimi
- Responsive tasarım

- SQLite veritabanı desteği 



