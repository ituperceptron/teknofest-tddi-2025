def summarize_text(llama_model, text, max_tokens=200):
    """Summarize text using LLaMA"""
    
    prompt = f"""<|start_header_id|>system<|end_header_id|>

    Sen Türkçe metinleri özetleyen bir asistansın. Verilen metni kısa ve öz bir şekilde özetle.

    <|eot_id|><|start_header_id|>user<|end_header_id|>

    Aşağıdaki metni özetle:

    {text}

    <|eot_id|><|start_header_id|>assistant<|end_header_id|>

    """
        
    try:
        response = llama_model(
            prompt,
            max_tokens=max_tokens,
            temperature=0.7,
            top_p=0.9,
            stop=["<|eot_id|>"],
            echo=False
        )
        
        summary = response['choices'][0]['text'].strip()
        
        return {
            'success': True,
            'summary': summary,
            'original_length': len(text),
            'summary_length': len(summary)
        }
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }