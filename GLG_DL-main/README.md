# GLG_DL
NAMED ENTITY RECOGNITION:
- Copy the NER.py script to the project
- Copy the finetuned BERT model files from here:
    * https://drive.google.com/file/d/1ZElGgNePMQMwMiepsOx0MkGWDMr-kQjM/view?usp=sharing
    * https://drive.google.com/file/d/1CRzkhZbcw4yT7vVvMmUuSyeJDRCBq76t/view?usp=sharing
    * https://drive.google.com/file/d/1Q5j4rWPWHK5iKeW-D8hWNBVctAavbce_/view?usp=sharing


- Run NER as follows:
```python
  pipeline = NER_pipeline()
  pipeline.run_ner(sample_text)
```

TOPIC MODELING:
 - The following pretrained files need to be downloaded before running main.py
 - The script outputs a Topic, and associated keywords for each sample text provided in main.py
 
TOpic Keywords:
https://drive.google.com/file/d/1-1LXRgomfpqloQEvbDgfR_Hx7SAnOfWD/view?usp=sharing

Topic Labels:
https://drive.google.com/file/d/1-3p3vZYdY8iytnDn5gUaUVcabKfkrUhq/view?usp=sharing

KMeans Model:
https://drive.google.com/file/d/1qBL6k2w06IbXs1h_tNJb52QW6gkKhBGy/view?usp=sharing

UMap Model:
https://drive.google.com/file/d/1pm6Vn09n8M_WZHyDwFJ_CiHLqMBR1taw/view?usp=sharing

        
        
