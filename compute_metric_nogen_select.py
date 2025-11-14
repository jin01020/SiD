import torch
import os
import glob
from tqdm import tqdm
from utils import (
    parse_arguments,  # Make sure this imports your parser
)
#from models import prepare_condition_loader, prepare_stuff
import time
import numpy as np
import PIL.Image
import argparse
import ast
import json
import sys
from PIL import Image
from typing import List, Dict, Any, Optional, Tuple

# HPS v2, Aesthetic Score, CLIP Score
try:
    import hpsv2  # HPS v2 라이브러리
except ImportError:
    print("Warning: hpsv2 not installed. HPS evaluation will be skipped.")
    hpsv2 = None

try:
    from transformers import AutoProcessor, AutoModel  # Aesthetic Score용
except ImportError:
    print("Warning: transformers not installed. Aesthetic evaluation will be skipped.")
    AutoProcessor, AutoModel = None, None

try:
    from torchmetrics.functional.multimodal import clip_score  # CLIP Score용
except ImportError:
    print("Warning: torchmetrics not installed. CLIP evaluation will be skipped.")
    clip_score = None

# --- FID Imports ---
try:
    import dnnlib
    import pickle
    import scipy
    from torch.nn.functional import adaptive_avg_pool2d
    from pytorch_fid.inception import InceptionV3
    FID_IMPORTS_AVAILABLE = True
except ImportError:
    print("Warning: FID-related libraries (dnnlib, pytorch_fid, scipy) not fully installed. FID evaluation will be skipped.")
    FID_IMPORTS_AVAILABLE = False


def setup_evaluators(device: str) -> Dict[str, Any]:
    """
    평가에 필요한 무거운 모델들을 미리 로드합니다. (Aesthetic)
    """
    if AutoProcessor is None:
        print("Aesthetic Score (transformers) 라이브러리가 없어 로드를 건너뜁니다.")
        return {"aes_model": None, "aes_processor": None}

    print("--- 평가 모델 로드 중 (Aesthetic Score) ---")
    try:
        model_id = "shunk031/aesthetics-predictor-v2-sac-logos-ava1-l14-linearMSE"
        processor = AutoProcessor.from_pretrained(model_id)
        model = AutoModel.from_pretrained(model_id, trust_remote_code=True).to(device)
        model.eval()
        print("Aesthetic Score 모델 로드 완료.")
        return {"aes_model": model, "aes_processor": processor}
    except Exception as e:
        print(f"Aesthetic Score 모델 로드 중 오류 발생: {e}")
        return {"aes_model": None, "aes_processor": None}

def compute_hps(pil_images: List[Image.Image], prompts: List[str]) -> Optional[List[float]]:
    """
    HPS v2 점수를 계산합니다. (개별적으로 루프)
    """
    if hpsv2 is None:
        print("HPS v2 라이브러리가 없어 계산을 건너뜁니다.")
        return None
        
    scores = []
    try:
        if len(pil_images) != len(prompts):
            print(f"HPS v2 Error: Mismatch in image ({len(pil_images)}) and prompt ({len(prompts)}) count.")
            return None

        for img, prompt in zip(pil_images, prompts):
            # hpsv2.score는 [score, ...,] 리스트를 반환
            result_list = hpsv2.score(img, prompt, hps_version="v2.1")
            scores.append(result_list[0])
        return scores
    except Exception as e:
        print(f"HPS v2 계산 중 오류 발생: {e}")
        return None


# def compute_hps(pil_images: List[Image.Image], prompts: List[str]) -> Optional[List[float]]:
#     """
#     HPS v2 점수를 계산합니다. (배치 처리로 수정됨)
#     """
#     if hpsv2 is None:
#         print("HPS v2 라이브러리가 없어 계산을 건너뜁니다.")
#         return None
        
#     try:
#         if len(pil_images) != len(prompts):
#             print(f"HPS v2 Error: Mismatch in image ({len(pil_images)}) and prompt ({len(prompts)}) count.")
#             return None

#         # --- 수정된 부분 ---
#         # for 루프 대신, 이미지 리스트와 프롬프트 리스트를 통째로 전달
#         # hpsv2.score가 [[score1, ...], [score2, ...]] 형태의 리스트를 반환
#         result_lists = hpsv2.score(
#             pil_images, 
#             prompts, 
#             hps_version="v2.1"
#         )
        
#         # 각 결과 리스트에서 첫 번째 값(점수)만 추출
#         scores = [result[0] for result in result_lists]
#         # --- 수정 완료 ---
        
#         return scores
        
#     except Exception as e:
#         print(f"HPS v2 (Batch) 계산 중 오류 발생: {e}")
#         return None


def compute_aesthetic(
    pil_images: List[Image.Image], 
    aes_model: torch.nn.Module, 
    aes_processor: Any, 
    device: str
) -> Optional[List[float]]:
    """Aesthetic Score를 계산합니다. (배치 처리)"""
    if aes_model is None or aes_processor is None:
        print("Aesthetic Score 모델이 로드되지 않아 계산을 건너뜁니다.")
        return None
        
    try:
        with torch.no_grad():
            inputs = aes_processor(images=pil_images, return_tensors="pt").to(device)
            outputs = aes_model(**inputs)
            scores = outputs.logits.flatten().tolist()
        return scores
    except Exception as e:
        print(f"Aesthetic Score 계산 중 오류 발생: {e}")
        return None

def compute_clip(
    pil_images: List[Image.Image], 
    prompts: List[str], 
    device: str
) -> Optional[float]:
    """CLIP Score의 배치 평균을 계산합니다. (TorchMetrics 사용)"""
    if clip_score is None:
        print("CLIP Score (torchmetrics) 라이브러리가 없어 계산을 건너뜁니다.")
        return None
        
    try:
        image_tensors = []
        target_size = pil_images[0].size 
        
        for img in pil_images:
            if img.size != target_size:
                # PIL.Image.Resampling.LANCZOS 사용
                img = img.resize(target_size, Image.Resampling.LANCZOS)
            img_array = np.array(img)
            tensor_uint8 = torch.from_numpy(img_array).permute(2, 0, 1) # (C, H, W)
            image_tensors.append(tensor_uint8)

        images_batch = torch.stack(image_tensors).to(device)

        score = clip_score(
            images_batch, 
            prompts, 
            model_name_or_path="openai/clip-vit-large-patch14"
        ).item()
        return score
    except Exception as e:
        print(f"CLIP Score 계산 중 오류 발생: {e}")
        return None

# --- MODIFIED FUNCTION ---
def evaluate_images(
    pil_images: List[Image.Image], 
    prompts: List[str],
    evaluators: Dict[str, Any], 
    device: str,
    eval_hps: bool,
    eval_aes: bool,
    eval_clip: bool
) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """
    HPS, Aesthetic, CLIP의 배치 평균 점수들을 계산합니다.
    (플래그에 따라 선택적으로 계산)
    """
    avg_hps = None
    avg_aes = None
    avg_clip = None

    # 1. HPS v2 (returns List[float])
    if eval_hps:
        hps_scores_list = compute_hps(pil_images, prompts)
        avg_hps = np.mean(hps_scores_list) if hps_scores_list else None
    
    # 2. Aesthetic Score (returns List[float])
    if eval_aes:
        aes_scores_list = compute_aesthetic(
            pil_images, 
            evaluators.get("aes_model"), 
            evaluators.get("aes_processor"), 
            device
        )
        avg_aes = np.mean(aes_scores_list) if aes_scores_list else None
    
    # 3. CLIP Score (returns float)
    if eval_clip:
        avg_clip = compute_clip(pil_images, prompts, device)
    
    return avg_hps, avg_aes, avg_clip
# --- END MODIFIED FUNCTION ---


# --- NEW FID HELPER FUNCTION ---
def calculate_fid_from_inception_stats(mu, sigma, mu_ref, sigma_ref):
    m = np.square(mu - mu_ref).sum()
    s, _ = scipy.linalg.sqrtm(np.dot(sigma, sigma_ref), disp=False)
    fid = m + np.trace(sigma + sigma_ref - s * 2)
    return float(np.real(fid))
# --- END NEW FUNCTION ---


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # --- 1. Setup Aesthetic/CLIP/HPS Evaluators (Conditional) ---
    evaluator_models = {}
    if args.eval_aes:
        evaluator_models = setup_evaluators(device)
    
    # --- 2. Setup FID Models (Conditional) ---
    fid_model = None
    detector_net = None
    ref = None
    FEATURE_DIM = 2048 # FID_IMPORTS_AVAILABLE이 True일 때만 사용됨

    if args.eval_fid:
        if not FID_IMPORTS_AVAILABLE:
            print("Error: FID 라이브러리가 설치되지 않아 FID 계산을 중단합니다.")
            args.eval_fid = False # 강제로 끔
        else:
            print("Loading FID models and reference stats...")
            if not hasattr(args, 'ref_path') or not args.ref_path:
                print("Error: --ref_path argument is required for FID.")
                print("Disabling FID evaluation.")
                args.eval_fid = False
            else:
                try:
                    # Load pytorch-fid (LD-style) model
                    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[FEATURE_DIM]
                    fid_model = InceptionV3([block_idx]).to(device)
                    fid_model.eval()
                    
                    # Load EDM-style model
                    DETECTOR_URL = "https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/metrics/inception-2015-12-05.pkl"
                    with dnnlib.util.open_url(DETECTOR_URL, verbose=False) as f:
                        detector_net = pickle.load(f).to(device)

                    # Load reference stats
                    with dnnlib.util.open_url(args.ref_path) as f:
                        ref = dict(np.load(f, allow_pickle=True))
                    print("FID models and reference stats loaded.")
                except Exception as e:
                    print(f"FID 모델 로드 중 오류 발생: {e}. FID 평가를 비활성화합니다.")
                    args.eval_fid = False

    # --- 3. Load Prompts ---
    # (프롬프트는 HPS/CLIP 또는 이미지 매칭에 필요할 수 있으므로 항상 로드)
    """
    wrapped_model, model, decoding_fn, noise_schedule, latent_resolution, latent_channel, _, _, encoding_fn = prepare_stuff(args)
    condition_loader = prepare_condition_loader(model_type=args.model, 
                                                model=model,
                                                scale=args.scale if hasattr(args, "scale") else None,
                                                condition=args.prompt_path or "uniform", 
                                                sampling_batch_size=args.sampling_batch_size,
                                                num_prompt=None,
                                                )
    all_prompts = condition_loader.prompts
    if args.prompt_path is not None:
        print(f"Loaded {len(all_prompts)} prompts from {args.prompt_path}")

    # --- 4. Discover Image Files ---
    if not hasattr(args, 'data_dir') or not args.data_dir:
        print("Error: --data_dir argument is required.")
        sys.exit(1)
    """    
    data_dir = args.data_dir
    image_files = sorted(glob.glob(os.path.join(data_dir, "*.png")))
    
    if not image_files:
        print(f"Error: No .png files found in {data_dir}")
        sys.exit(1)
    
    print(f"Found {len(image_files)} images in {data_dir}")

    # --- 5. Match Prompts to Images ---
    num_images = len(image_files)
    #num_prompts = len(all_prompts)

    # if num_images > num_prompts:
    #     print(f"Warning: Found {num_images} images but only {num_prompts} prompts. Truncating to {num_prompts} images.")
    #     image_files = image_files[:num_prompts]
    # elif num_prompts > num_images:
    #     print(f"Warning: Found {num_images} images but {num_prompts} prompts. Truncating to {num_images} prompts.")
    #     all_prompts = all_prompts[:num_images]
        
    num_to_process = len(image_files)
    
    # if hasattr(args, 'total_samples') and args.total_samples < num_to_process:
    #     print(f"Limiting to first {args.total_samples} samples as requested.")
    #     num_to_process = args.total_samples
    #     image_files = image_files[:num_to_process]
    #     all_prompts = all_prompts[:num_to_process]

    # --- 6. Setup Batching & Dirs ---
    if not hasattr(args, 'sampling_batch_size'):
        print("Warning: --sampling_batch_size not in parse_arguments. Defaulting to 4.")
        args.sampling_batch_size = 4
    
    print(f"Processing {num_to_process} images in batches of {args.sampling_batch_size}")
    output_dir = os.path.join(args.data_dir, "metric_results")
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "command_line.txt"), "w") as f:
        f.write(command_line + "\n")

    start = time.time()
    
    # HPS/Aesthetic/CLIP accumulators (list of batch averages)
    all_batch_hps_scores = []
    all_batch_aes_scores = []
    all_batch_clip_scores = []
    
    # FID accumulators (Conditional)
    mu_edm = None
    sigma_edm = None
    act_arr = None
    fid_start_idx = 0

    if args.eval_fid:
        mu_edm = torch.zeros([FEATURE_DIM], dtype=torch.float64, device=device)
        sigma_edm = torch.zeros([FEATURE_DIM, FEATURE_DIM], dtype=torch.float64, device=device)
        act_arr = np.empty((num_to_process, FEATURE_DIM)) # For pytorch-fid
        fid_start_idx = 0


    num_batches = (num_to_process + args.sampling_batch_size - 1) // args.sampling_batch_size

    # --- 7. Main BATCHED Evaluation Loop ---
    for batch_idx in tqdm(range(num_batches), desc="Evaluating batches"):
        start_idx = batch_idx * args.sampling_batch_size
        end_idx = min((batch_idx + 1) * args.sampling_batch_size, num_to_process)

        batch_image_paths = image_files[start_idx:end_idx]
        #batch_prompts = all_prompts[start_idx:end_idx]
        
        batch_pil_images = []
        valid_prompts = []
        
        # Load PIL images for the batch
        for i, img_path in enumerate(batch_image_paths):
            try:
                batch_pil_images.append(PIL.Image.open(img_path).convert("RGB"))
                #valid_prompts.append(batch_prompts[i]) # Only add prompt if image loads
            except Exception as e:
                print(f"Warning: Failed to load {img_path}, skipping. Error: {e}")
        
        if not batch_pil_images:
            print(f"Skipping empty batch {batch_idx}")
            continue

        try:
            # --- 7a. HPS, Aesthetic, CLIP Eval (Conditional) ---
            if args.eval_hps or args.eval_aes or args.eval_clip:
                avg_hps, avg_aes, avg_clip = evaluate_images(
                    batch_pil_images, 
                    valid_prompts,
                    evaluators=evaluator_models,
                    device=device,
                    eval_hps=args.eval_hps,
                    eval_aes=args.eval_aes,
                    eval_clip=args.eval_clip
                )
                
                if avg_hps is not None: all_batch_hps_scores.append(avg_hps)
                if avg_aes is not None: all_batch_aes_scores.append(avg_aes)
                if avg_clip is not None: all_batch_clip_scores.append(avg_clip)

            # --- 7b. FID Feature Extraction (Conditional) ---
            if args.eval_fid:
                tensors_for_ld = []  # float [0, 1]
                tensors_for_edm = [] # uint8 [0, 255]
                
                for img in batch_pil_images:
                    # Convert PIL [0, 255] to Tensors
                    img_array = np.array(img) # (H, W, C)
                    tensor_uint8 = torch.from_numpy(img_array).permute(2, 0, 1) # (C, H, W)
                    tensors_for_edm.append(tensor_uint8)
                    tensors_for_ld.append(tensor_uint8.float() / 255.0) # Convert to [0, 1]
                
                batch_tensors_edm = torch.stack(tensors_for_edm).to(device)
                batch_tensors_ld = torch.stack(tensors_for_ld).to(device)
                
                # EDM/StyleGAN FID features
                features_edm = detector_net(batch_tensors_edm, return_features=True).to(torch.float64)
                mu_edm += features_edm.sum(0)
                sigma_edm += features_edm.T @ features_edm

                # pytorch-fid (LD-style) features
                with torch.no_grad():
                    pred_ld = fid_model(batch_tensors_ld)[0]
                
                if pred_ld.size(2) != 1 or pred_ld.size(3) != 1:
                    pred_ld = adaptive_avg_pool2d(pred_ld, output_size=(1, 1))
                
                pred_ld = pred_ld.squeeze(3).squeeze(2).cpu().numpy()
                act_arr[fid_start_idx : fid_start_idx + pred_ld.shape[0]] = pred_ld
                fid_start_idx += pred_ld.shape[0]

        except Exception as e:
            print(f"\nError processing batch {batch_idx}: {e}")

    end = time.time()
    print(f"\n--- Evaluation Finished in {end - start:.2f} seconds ---")

    # --- 8. Final Calculations & Summary ---
    print("\n--- Final Average Scores ---")
    
    # HPS/Aesthetic/CLIP
    # (리스트가 비어있으면 np.mean이 RuntimeWarning을 내지만, 
    #  ... if all_batch..._scores else "N/A" 구문이 이를 방지)
    final_avg_hps = np.mean(all_batch_hps_scores) if all_batch_hps_scores else "N/A"
    final_avg_aes = np.mean(all_batch_aes_scores) if all_batch_aes_scores else "N/A"
    final_avg_clip = np.mean(all_batch_clip_scores) if all_batch_clip_scores else "N/A"
    
    if args.eval_hps: print(f"  Avg HPSv2: {final_avg_hps if isinstance(final_avg_hps, str) else f'{final_avg_hps:.4f}'}")
    if args.eval_aes: print(f"  Avg Aesthetic: {final_avg_aes if isinstance(final_avg_aes, str) else f'{final_avg_aes:.4f}'}")
    if args.eval_clip: print(f"  Avg CLIP: {final_avg_clip if isinstance(final_avg_clip, str) else f'{final_avg_clip:.4f}'}")
    
    # FID (Conditional)
    fid_edm = "N/A"
    fid_latent_diff = "N/A"

    if args.eval_fid:
        print("Calculating FID scores...")
        if fid_start_idx != num_to_process:
            print(f"Warning: Processed {fid_start_idx} images for FID, but expected {num_to_process}. Truncating.")
            act_arr = act_arr[:fid_start_idx]
            # mu_edm, sigma_edm은 num_to_process 기준으로 정규화되므로 괜찮음
        
        try:
            # EDM FID
            mu_edm /= num_to_process
            sigma_edm -= mu_edm.ger(mu_edm) * num_to_process
            sigma_edm /= num_to_process - 1
            mu_edm = mu_edm.cpu().numpy()
            sigma_edm = sigma_edm.cpu().numpy()
            fid_edm = calculate_fid_from_inception_stats(mu_edm, sigma_edm, ref["mu"], ref["sigma"])
            print(f"  FID (EDM): {fid_edm:.4f}")

            # LD / pytorch-fid
            mu_ld = np.mean(act_arr, axis=0)
            sigma_ld = np.cov(act_arr, rowvar=False)
            fid_latent_diff = calculate_fid_from_inception_stats(mu_ld, sigma_ld, ref["mu"], ref["sigma"])
            print(f"  FID (LD/pytorch-fid): {fid_latent_diff:.4f}")
            
        except Exception as e:
            print(f"Could not calculate FID: {e}")
            fid_edm = "N/A" # 이미 N/A지만 명시적으로
            fid_latent_diff = "N/A"
    
    print(f"  Total Samples Evaluated: {num_to_process}")
    
    final_averages = {
        "experiment_name": os.path.basename(args.data_dir.strip('/')),
        "avg_hps_score": final_avg_hps,
        "avg_aesthetic_score": final_avg_aes,
        "avg_clip_score": final_avg_clip,
        "fid_edm": fid_edm,
        "fid_latent_diff": fid_latent_diff,
        "num_samples": num_to_process
    }

    # --- 9. Save Summary ---
    summary_path = os.path.join(output_dir, "evaluation_summary.json")
    # 1. 기존 결과 읽기
    results_list = []
    if os.path.exists(summary_path):
        try:
            with open(summary_path, "r", encoding="utf-8") as f:
                results_list = json.load(f)
                if not isinstance(results_list, list): # 혹시 파일이 깨졌으면
                    results_list = []
        except json.JSONDecodeError:
            results_list = [] # 파일이 비어있거나 깨졌으면 새로 시작
    
    # 2. 새 결과 추가
    results_list.append(final_averages)
    
    # 3. 전체 리스트를 'w' 모드로 덮어쓰기
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(results_list, f, indent=2, default=str)

    print(f"\nSaved summary (updated list) to {summary_path}")


if __name__ == "__main__":
    command_line = "python " + " ".join(sys.argv)
    args = parse_arguments()
    
    # --- 중요 ---
    # utils.py의 parse_arguments() 함수에 아래 인자들을 추가해야 합니다.
    # (기존 인자와 충돌하지 않게 확인하세요)
    #
    # 예시 (ArgumentParser 인스턴스를 parser라고 가정):
    #
    # parser.add_argument("--eval_hps", action="store_true", help="Enable HPS v2 evaluation.")
    parser.add_argument("--eval_aes", action="store_true", help="Enable Aesthetic Score evaluation.")
    # parser.add_argument("--eval_clip", action="store_true", help="Enable CLIP Score evaluation.")
    # parser.add_argument("--eval_fid", action="store_true", help="Enable FID evaluation.")
    #
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing generated images.")
    # parser.add_argument("--ref_path", type=str, default=None, help="Path to reference stats for FID (e.g., .npz file). Required if --eval_fid is set.")
    # parser.add_argument("--total_samples", type=int, default=float('inf'), help="Limit evaluation to this many samples.")
    #
    # (sampling_batch_size는 이미 있는 것 같지만, 없다면 추가)
    parser.add_argument("--sampling_batch_size", type=int, default=16, help="Batch size for evaluation loop.")
    # ---

    # (args에 eval_... 플래그가 없으면 False로 기본값 설정)
    if not hasattr(args, "eval_hps"): args.eval_hps = False
    if not hasattr(args, "eval_aes"): args.eval_aes = False
    if not hasattr(args, "eval_clip"): args.eval_clip = False
    if not hasattr(args, "eval_fid"): args.eval_fid = False

    if not (args.eval_hps or args.eval_aes or args.eval_clip or args.eval_fid):
        print("모든 평가가 비활성화되었습니다. (e.g., --eval_hps 사용). 종료합니다.")
        sys.exit(0)

    main(args)