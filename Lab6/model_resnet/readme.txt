--- Model training ---

Run the model training code with
	python3 model_trainer.py

This will generate a .pth file in the folder model_parameters with the trained weights. Small variation because of random split.


--- Grading ---
(subfolder 2025S_imgs with images and label.txt file should exist)
Run with
	python3 model_grader . py −−data_path ./2025S_imgs −−model_path ./resnet18_parameters_best.pth