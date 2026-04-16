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
I will send you more data to see how much performance is affected, but this is a good starting point. The modification
of hyperparameters is at the top of the file at line 34 in class Config. Change the parameters around and record the
results which we will put in the final report. 

-> Additional task: Try to see if you can add a classification head on top of the model architecture and then finetune.
You won't be able to do this on Tinker as it does not support. This would like our homeworks where you access model weights
through hugginface and then add the classification head and then train on OSC. Try to see if you can do this, it would be 
very nice to have in the final report. But from what i have seen getting time on OSC is hard. 

