activate: 
	@echo "Activating virtual environment..."
	@python3 -m venv llm_env
	@. llm_env/bin/activate
	@pip install --upgrade pip
	@pip install -r requirements.txt
	@echo "Virtual environment activated."