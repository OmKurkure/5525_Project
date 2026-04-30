-> First create a python environment (Python version 3.11): 
python3.11 -m venv .venv311
source .venv311/bin/activate

-> Then install the requirements
pip install --upgrade pip
pip install -r requirements.txt

-> Go to Tinker console and create an API key (https://tinker-console.thinkingmachines.ai/). Copy the key and then export Tinker API key in terminal
export TINKER_API_KEY="your-api-key-here"

-> Go to huggingface and get access to the Llama-3.2-1B model: https://huggingface.co/meta-llama/Llama-3.2-1B, you should 
get access to the model after some time (it won't be too long, like max 1 hour)

-> The data we are using is in the data folder. 400 train and 100 test examples to simulate low resource conditions. 

-> python3 train.py to train. Hyperparameters can be changed in the config at the top of the train.py script. 
