from gllm import LLM
import argparse



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Chat with llama')
    parser.add_argument("--model-path",type=str,required=True)
    args = parser.parse_args()
    
    model_path = '/mnt/sda/2022-0526/home/gtyinstinct/.cache/modelscope/hub/LLM-Research/Meta-Llama-3-8B-Instruct'
    llm = LLM(args.model_path)
    llm.chat()
    