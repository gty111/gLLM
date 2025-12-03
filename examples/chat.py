from gllm import LLM
import argparse



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Chat with LLM')
    parser.add_argument("--model",type=str,required=True)
    parser.add_argument('--pp', type=int, default=1)
    parser.add_argument('--tp', type=int, default=1)
    parser.add_argument('--master-port', type=str, default='8000')
    parser.add_argument('--enable-cuda-graph', help='Enable cuda graph', action='store_true')
    args = parser.parse_args()
    
    llm = LLM(args.model,
              pp_size=args.pp,
              tp_size=args.tp,
              master_port=args.master_port,
              enable_cuda_graph=args.enable_cuda_graph,
              gpu_memory_util=0.8)
    llm.chat()
    