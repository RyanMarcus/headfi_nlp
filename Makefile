
models/lcd_model.pkl: models/training_data.pkl lcd_impressions.txt nlp.py
	bash -c "time python3 nlp.py train"

models/training_data.pkl: lcd_impressions.txt 
	bash -c "time python3 nlp.py extract"


.phony: eval clean 

eval: lcd_eval.txt models/training_data.pkl models/lcd_model.pkl nlp.py
	python3 nlp.py test

clean:
	rm -rf models/*
