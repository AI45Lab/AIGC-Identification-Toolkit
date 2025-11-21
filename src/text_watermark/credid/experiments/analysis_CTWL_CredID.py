from src.utils.wm_lm import WmLMArgs
from watermarking.arg_classes.wm_arg_class import WmAnalysisArgs
from transformers import HfArgumentParser
from src.utils.wm_analysis import main

oracle_model_name = 'huggyllama/llama-13b'
parser = HfArgumentParser((WmLMArgs,))
args: WmLMArgs
args, = parser.parse_args_into_dataclasses()
# save_file_path = args.complete_save_file_path
save_file_path = "my_watermark_result/lm_new_7_10/huggyllama-llama-7b_1.5_1.5_300_400_200_42_42_10_40.0_4_1.0_10_-1_300_max_confidence_updated_0.5_huggyllama/llama-7b_1000_nofeedback.json"

analysis_args = WmAnalysisArgs(analysis_file_name=save_file_path, device=args.device,
                      oracle_model_name=oracle_model_name, args=args)
main(analysis_args)
