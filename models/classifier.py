def classify_text(classifier_pipeline, text, categories=None):
    """Classify text using XLM-RoBERTa"""
    
    if categories is None:
        categories = [
            "Şikayet dilekçesi",
            "Bilgi edinme başvurusu",
            "Sosyal yardım talebi",
            "İzin ruhsat başvurusu",
            "Vergi maliye işlemi",
            "Atama nakil talebi",
            "Ödeme makbuzu",
            "Kanun görüş önerisi",
            "Duyuru"
        ]
    
    try:
        result = classifier_pipeline(text, categories)
        
        # Format results
        classification_results = []
        for label, score in zip(result['labels'], result['scores']):
            classification_results.append({
                'name': label,
                'confidence': float(score)
            })
        
        return {
            'success': True,
            'categories': classification_results,
            'main_category': classification_results[0]['name'],
            'confidence': classification_results[0]['confidence']
        }
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }