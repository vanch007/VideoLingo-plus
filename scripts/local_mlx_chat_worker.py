"""JSON-lines worker for an installed local MLX language model; no downloads."""
import argparse
import contextlib
import json
from pathlib import Path
import sys


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--model-path",required=True)
    args=parser.parse_args()
    model_path=Path(args.model_path)
    if not model_path.is_dir() or not (model_path/"config.json").is_file():
        raise FileNotFoundError("Expected already installed local model weights")
    from mlx_lm import load,stream_generate
    from mlx_lm.sample_utils import make_sampler
    with contextlib.redirect_stdout(sys.stderr):
        model,tokenizer=load(str(model_path),tokenizer_config={"trust_remote_code":False})
    print(json.dumps({"ready":True,"model_path":str(model_path)}),flush=True)
    for raw in sys.stdin:
        try:
            request=json.loads(raw)
            messages=request["messages"]
            kwargs=request.get("extra_body",{}).get("chat_template_kwargs",{})
            prompt=tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,**kwargs)
            output=[];last=None
            with contextlib.redirect_stdout(sys.stderr):
                for response in stream_generate(model,tokenizer,prompt,
                        max_tokens=int(request.get("max_tokens",4096)),
                        sampler=make_sampler(temp=float(request.get("temperature",0)))):
                    output.append(response.text);last=response
            if last is None or not last.finish_reason:
                raise RuntimeError("Local model returned no completed generation")
            result={"content":"".join(output),"finish_reason":last.finish_reason,
                    "usage":{"prompt_tokens":last.prompt_tokens,"completion_tokens":last.generation_tokens,
                             "total_tokens":last.prompt_tokens+last.generation_tokens}}
        except Exception as exc:
            result={"error":f"{type(exc).__name__}: {exc}"}
        print(json.dumps(result,ensure_ascii=False),flush=True)


if __name__=="__main__":
    main()
