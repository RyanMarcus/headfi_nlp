
models/lcd_model.pkl: models/training_data.pkl  nlp.py
	bash -c "time python3 nlp.py train"

models/training_data.pkl: lcd_impressions.txt he400_impressions.txt
	bash -c "time python3 nlp.py extract"


.phony: eval clean train

eval: lcd_eval.txt models/training_data.pkl models/lcd_model.pkl nlp.py
	python3 nlp.py test

train: models/training_data.pkl
	python3 nlp.py train

clean:
	rm -rf models/*
