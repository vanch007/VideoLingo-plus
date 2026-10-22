import sys
import os
import json
import argparse
from pathlib import Path

# Add skill path to sys.path
sys.path.insert(0, "/Users/vanch/.gemini/antigravity/skills/ocr-recognition/scripts")
import ocr_local

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    args = parser.parse_args()

    model_dir = args.model
    model, processor = ocr_local.load_local_model(model_dir, None)
    
    # Notify ready on fd 3 or stdout
    print(json.dumps({"status": "ready"}), flush=True)

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            if req.get("action") == "stop":
                break
            
            image_path = Path(req.get("image_path"))
            task = req.get("task", "boxes")
            prompt = ocr_local.build_prompt(task, req.get("prompt"))

            res_doc = ocr_local.recognize_document(
                image_path,
                model,
                processor,
                model_dir,
                None,
                task,
                prompt,
                max_tokens=2048,
                dpi=160,
                start_page=1,
                end_page=1
            )

            regions = []
            raw_text = ""
            if res_doc.get("pages"):
                page = res_doc["pages"][0]
                regions = page.get("regions", [])
                raw_text = page.get("text", "")

            res = {
                "status": "success",
                "image_path": str(image_path),
                "text": raw_text,
                "regions": regions
            }
            print(json.dumps(res, ensure_ascii=False), flush=True)
        except Exception as e:
            res = {
                "status": "error",
                "error": str(e),
                "image_path": req.get("image_path") if 'req' in locals() else None
            }
            print(json.dumps(res, ensure_ascii=False), flush=True)

if __name__ == "__main__":
    main()

