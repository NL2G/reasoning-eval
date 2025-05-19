import re
from argparse import ArgumentParser
from pathlib import Path
import json
import pandas as pd
from collections import Counter

import sienna
from scipy.stats import pearsonr
from reasoning_eval.summarization_evaluation.utils import DATA_BASE_DIR

# Attempt to import transformers, but make it optional if not analyzing deepseek models
try:
    from transformers import AutoTokenizer
except ImportError:
    AutoTokenizer = None

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

def aggreage_expert_annotations(data_point: dict) -> dict:
    aspects = ["coherence", "consistency", "fluency", "relevance"]
    aspect_score = {aspect: -1 for aspect in aspects}
    for aspect in aspects:
        scores = []
        for annotation in data_point["expert_annotations"]:
            scores.append(annotation[aspect])
        aspect_score[aspect] = sum(scores) / len(scores) if scores else -1 # Avoid division by zero
    return aspect_score

def parse_output(output_data: dict[str, str]) -> int | float | None:
    content = output_data.get("content", "")
    if not content:
        return None
        
    # Attempt to match formats like "Coherence: 4" or "- Coherence: 4" or just "4"
    # Prioritize stricter formats first
    metric_keywords = ["Coherence", "Consistency", "Fluency", "Relevance"] # Add other metrics if needed
    
    # Try to find a score associated with a known metric keyword if present
    for keyword in metric_keywords:
        if keyword in content:
            # Search for a digit possibly after the keyword and a colon/space
            score_match = re.search(rf"{keyword}[-:\s]*(\d)", content, re.IGNORECASE)
            if score_match and score_match.group(1):
                try:
                    return int(score_match.group(1))
                except ValueError:
                    continue # continue to next keyword or general search
            # If keyword is present but format is just a number after it (e.g. "Coherence\\n\\n3")
            # This is a bit more risky, try to find number in proximity
            keyword_idx = content.find(keyword)
            search_area = content[keyword_idx:]
            score_match_prox = re.search(r"(\d)", search_area, re.IGNORECASE)
            if score_match_prox:
                 try:
                    return int(score_match_prox.group(0))
                 except ValueError:
                    continue


    # If no keyword match or keyword-associated score found, try to find any standalone digit.
    # This handles cases where content is just "3" or similar.
    # To make it more robust, we can look for digits at the end of the string or surrounded by common delimiters.
    score_match_general = re.search(r"(\d)", content, re.IGNORECASE)
    if score_match_general:
        try:
            return int(score_match_general.group(0))
        except ValueError:
            return None # Cannot parse
            
    return None


def get_reasoning_length(prediction_item: dict, model_type: str, tokenizer_name: str | None = None) -> int | None:
    if model_type == "openai":
        try:
            return int(prediction_item["usage"]["completion_tokens_details"]["reasoning_tokens"])
        except (KeyError, TypeError, ValueError):
            try: # Fallback if reasoning_tokens specifically is missing but other token info exists
                # This is a heuristic: if reasoning_tokens is not available, 
                # and there's a "reasoning" field, we might try to tokenize it.
                # For now, returning None or 0 if the specific field is missing.
                # Or, one could argue 'completion_tokens' might be a proxy if 'reasoning_tokens' is absent.
                # Let's assume if 'reasoning_tokens' is not there, it's 0 for openai.
                if prediction_item.get("reasoning") is not None: # If there is a reasoning field
                    if prediction_item["reasoning"] == "": return 0 # Empty reasoning string
                    # If reasoning is present but no token count, it's tricky.
                    # For simplicity, if 'reasoning_tokens' is missing, we will return None to skip this data point for length analysis
                    # or one could default to 0. User context suggests specific field for openai.
                    return 0 # Default to 0 if field exists but no reasoning_tokens
                return 0 # Defaulting to 0 if reasoning_tokens path is broken or no reasoning content
            except Exception:
                return None # Truly unable to determine
    elif model_type == "deepseek":
        reasoning_text = prediction_item.get("reasoning", "")
        if not isinstance(reasoning_text, str) or not reasoning_text.strip():
            return 0
        # Count tokens by splitting reasoning text into words via regex
        words = re.findall(r"\w+", reasoning_text)
        return len(words)
    else:
        print(f"Unknown model type: {model_type}")
        return None

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--output-files", type=str, nargs='+', help="One or more LLM output files (.jsonl) to concatenate and evaluate")
    parser.add_argument("--metric-type", type=str, choices=["coherence", "consistency", "fluency", "relevance"], help="Metric to evaluate (e.g., coherence). Required unless generating CSV.")
    parser.add_argument("--model-type", type=str, required=True, choices=["openai", "deepseek"], help="Type of model (openai or deepseek)")
    parser.add_argument("--tokenizer-name", type=str, required=False, default="deepseek-ai/deepseek-coder-1.3b-instruct", help="Hugging Face tokenizer name for deepseek models (if not default)")
    parser.add_argument("--cut-off", type=int, required=False, default=None, help="Optional cut-off for the number of data points to process")
    parser.add_argument("--generate-plot", action="store_true", help="Generate a scatter plot of reasoning length vs. error.")
    parser.add_argument("--plot-output-path", type=str, default=None, help="Optional path to save the generated plot (e.g., 'plot.png'). If not provided and --generate-plot is set, tries to show plot.")
    parser.add_argument("--display-name", type=str, default=None, help="Custom display name for the model to use in plot titles and annotations.")
    parser.add_argument("--eval-dir", type=str, help="Directory containing JSONL files for a specific model and reasoning effort.")
    parser.add_argument("--model-name", type=str, help="Model name prefix for JSONL filenames (e.g., 'openai_o3-mini' or 'deepseek_deepseek-r1').")
    parser.add_argument("--reasoning-level", type=str, default=None, help="Reasoning effort level in filenames (e.g., 'high' for openai models).")
    parser.add_argument("--output-csv", type=str, help="Output CSV file path to write normalized data (id, aspect, normalized_llm_score, normalized_true_score, normalized_error, reasoning_length).")
    args = parser.parse_args()

    # Determine mode: CSV vs correlation
    csv_mode = False
    if args.output_csv:
        if not args.eval_dir or not args.model_name:
            parser.error("--output-csv requires --eval-dir and --model-name.")
        csv_mode = True
    else:
        if not args.output_files or not args.metric_type:
            parser.error("For correlation analysis, --output-files and --metric-type are required.")

    if not csv_mode:
        # Validate all provided output files for correlation mode
        output_file_paths = [Path(p) for p in args.output_files]
        for p in output_file_paths:
            if not p.exists():
                print(f"Error: Output file not found at {p}")
                exit(1)
    else:
        # Discover JSONL files for CSV mode
        eval_dir = Path(args.eval_dir)
        if not eval_dir.exists() or not eval_dir.is_dir():
            print(f"Error: Evaluation directory not found at {eval_dir}")
            exit(1)
        aspect_file_paths = []
        for p in eval_dir.iterdir():
            if p.suffix != ".jsonl":
                continue
            stem_parts = p.stem.split('.')
            if stem_parts[0] != args.model_name:
                continue
            if args.reasoning_level:
                if len(stem_parts) < 3 or stem_parts[1] != args.reasoning_level:
                    continue
                aspect = stem_parts[2]
            else:
                if len(stem_parts) != 2:
                    continue
                aspect = stem_parts[1]
            if aspect not in ["coherence", "consistency", "fluency", "relevance"]:
                continue
            aspect_file_paths.append((aspect, p))
        if not aspect_file_paths:
            print(f"No JSONL files found for model '{args.model_name}' with reasoning level '{args.reasoning_level}' in {eval_dir}")
            exit(1)

    # Load expert annotations data
    expert_data_path = DATA_BASE_DIR / "model_annotations.aligned.paired.jsonl"
    if not expert_data_path.exists():
        print(f"Error: Expert annotations file not found at {expert_data_path}")
        exit(1)
    
    expert_data = sienna.load(expert_data_path)
    
    # Build a mapping from example ID to expert data for lookup (allow repeated entries)
    expert_map = {dp.get("id"): dp for dp in expert_data if dp.get("id") is not None}
    
    # CSV mode processing: read, compute, normalize, and write CSV using pandas
    if csv_mode:
        df_list = []
        for aspect, p in aspect_file_paths:
            try:
                df = pd.read_json(p, lines=True)
            except ValueError as e:
                print(f"Error reading JSONL {p.name}: {e}")
                exit(1)
            df['aspect'] = aspect
            # Parse LLM score
            df['llm_score'] = df.apply(lambda row: parse_output(row.to_dict()), axis=1)
            df = df[df['llm_score'].between(1, 5)]
            # Expert score
            df['expert_score'] = df['id'].apply(lambda id: aggreage_expert_annotations(expert_map.get(id, {})).get(aspect, None))
            df = df[df['expert_score'].notna() & (df['expert_score'] != -1)]
            # Error and reasoning length
            df['error'] = (df['expert_score'] - df['llm_score']).abs()
            df['reasoning_length'] = df.apply(lambda row: get_reasoning_length(row.to_dict(), args.model_type, args.tokenizer_name), axis=1)
            df = df[df['reasoning_length'] > 0]
            df_list.append(df[['id', 'aspect', 'llm_score', 'expert_score', 'error', 'reasoning_length']])
        if not df_list:
            print("No valid records found for CSV mode.")
            exit(1)
        df_all = pd.concat(df_list, ignore_index=True)
        # Normalize per aspect
        for col, norm_col in [('llm_score', 'normalized_llm_score'), ('expert_score', 'normalized_true_score'), ('error', 'normalized_error')]:
            df_all[norm_col] = df_all.groupby('aspect')[col].transform(
                lambda x: (x - x.min()) / (x.max() - x.min()) if x.max() > x.min() else pd.Series(0, index=x.index)
            )
        # Write CSV
        df_out = df_all[['id', 'aspect', 'normalized_llm_score', 'normalized_true_score', 'normalized_error', 'reasoning_length']]
        df_out.to_csv(args.output_csv, index=False)
        print(f"Written normalized data to {args.output_csv}")
        exit(0)
    
    # Load and concatenate LLM predictions from all files
    llm_predictions = []
    for p in output_file_paths:
        try:
            with open(p, 'r') as f:
                for line in f:
                    try:
                        llm_predictions.append(json.loads(line))
                    except json.JSONDecodeError:
                        print(f"Warning: Could not decode JSON from line in {p.name}: {line.strip()}")
                        continue
        except Exception as e:
            print(f"Error loading LLM predictions from {p}: {e}")
            exit(1)

    if args.cut_off:
        llm_predictions = llm_predictions[:args.cut_off]

    reasoning_lengths = []
    errors = []
    llm_scores_list = []
    expert_scores_list = [] # For debugging or further analysis

    processed_count = 0
    skipped_count = 0

    # Process each prediction, allowing duplicate IDs
    for i, llm_pred_dp in enumerate(llm_predictions):
        # Validate prediction entry
        if not isinstance(llm_pred_dp, dict):
            print(f"Warning: Skipping prediction at index {i} due to unexpected data type: {type(llm_pred_dp)}")
            skipped_count +=1
            continue
        # Retrieve corresponding expert data by ID
        pred_id = llm_pred_dp.get("id")
        if pred_id is None:
            print(f"Warning: Skipping prediction at index {i} due to missing 'id' field")
            skipped_count +=1
            continue
        expert_dp = expert_map.get(pred_id)
        if expert_dp is None:
            print(f"Warning: No expert data found for id '{pred_id}', skipping.")
            skipped_count +=1
            continue

        # 1. Get Expert Score
        avg_expert_scores = aggreage_expert_annotations(expert_dp)
        expert_score = avg_expert_scores.get(args.metric_type)

        if expert_score is None or expert_score == -1: # -1 if no scores for aspect
            # print(f"Warning: Skipping data point {pred_id} due to missing expert score for metric '{args.metric_type}'.")
            skipped_count +=1
            continue

        # 2. Get LLM Predicted Score
        llm_score = parse_output(llm_pred_dp)
        if llm_score is None or not (1 <= llm_score <= 5): # Assuming scores are 1-5
            # print(f"Warning: Skipping data point {pred_id} due to invalid LLM score: {llm_score}.")
            skipped_count +=1
            continue
            
        # 3. Calculate Error
        error = abs(expert_score - llm_score)

        # 4. Get Reasoning Length
        reasoning_length = get_reasoning_length(llm_pred_dp, args.model_type, args.tokenizer_name)
        # Skip if no reasoning length or zero tokens
        if reasoning_length is None or reasoning_length <= 0:
            # print(f"Warning: Skipping data point {pred_id} due to zero or undefined reasoning length: {reasoning_length}.")
            skipped_count +=1
            continue

        reasoning_lengths.append(reasoning_length)
        errors.append(error)
        llm_scores_list.append(llm_score)
        expert_scores_list.append(expert_score)
        processed_count +=1
    
    print(f"Processed {processed_count} data points. Skipped {skipped_count} data points.")

    if not reasoning_lengths or len(reasoning_lengths) < 2: # Need at least 2 points for correlation
        print("Not enough data points to calculate correlations.")
        exit(0)

    # Calculate Correlations
    try:
        corr_length_error, p_length_error = pearsonr(reasoning_lengths, errors)
        corr_length_score, p_length_score = pearsonr(reasoning_lengths, llm_scores_list)
        
        print(f"Correlation between Reasoning Length and Error ({args.metric_type}):")
        print(f"  Pearson r: {corr_length_error:.4f} (p-value: {p_length_error:.4f})")
        print(f"Number of pairs for Length vs Error: {len(reasoning_lengths)}")

        print(f"\nCorrelation between Reasoning Length and LLM-Predicted Score ({args.metric_type}):")
        print(f"  Pearson r: {corr_length_score:.4f} (p-value: {p_length_score:.4f})")
        print(f"Number of pairs for Length vs LLM Score: {len(reasoning_lengths)}")

    except Exception as e:
        print(f"Error calculating correlations: {e}")

    # Generate and save/show plot if requested
    if args.generate_plot:
        if not MATPLOTLIB_AVAILABLE:
            print("\nWarning: Matplotlib is not installed. Cannot generate plot. Please install it: pip install matplotlib")
        elif not reasoning_lengths or not errors:
            print("\nWarning: No data available for plotting (reasoning_lengths or errors list is empty).")
        else:
            try:
                # Determine display name for plots/annotations (based on provided argument or file names)
                if args.display_name:
                    display_name = args.display_name
                else:
                    joined_names = "+".join([p.stem for p in output_file_paths])
                    display_name = f"{args.model_type}:{joined_names}"

                plt.figure(figsize=(10, 6))
                # Aggregate counts for identical (length, error) pairs to scale point sizes
                point_counts = Counter(zip(reasoning_lengths, errors))
                xs = []
                ys = []
                sizes = []
                for (x_val, y_val), count in point_counts.items():
                    xs.append(x_val)
                    ys.append(y_val)
                    sizes.append(count * 20)  # scale factor 20 for visibility; adjust if needed

                plt.scatter(xs, ys, s=sizes, alpha=0.5, edgecolors='k', linewidths=0.5)
                plt.title(f'{display_name} – Reasoning Length vs. Absolute Error ({args.metric_type})')
                plt.xlabel("Reasoning Length (tokens)")
                plt.ylabel(f"Absolute Error ({args.metric_type})")
                plt.grid(True)
                
                # Add Pearson correlation annotation to the plot
                plt.annotate(f"Pearson r: {corr_length_error:.4f} (p: {p_length_error:.4f})", 
                             xy=(0.05, 0.95), xycoords='axes fraction', 
                             bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
                
                if args.plot_output_path:
                    plt.savefig(args.plot_output_path)
                    print(f"\nPlot saved to {args.plot_output_path}")
                else:
                    print("\nDisplaying plot...")
                    plt.show()
            except Exception as e:
                print(f"\nError generating plot: {e}")
    
    # For further debugging or insight, you might want to see the distribution of lengths, errors, scores
    # import matplotlib.pyplot as plt
    # plt.figure(figsize=(12, 6))
    # plt.subplot(1, 2, 1)
    # plt.scatter(reasoning_lengths, errors)
    # plt.xlabel("Reasoning Length (tokens)")
    # plt.ylabel(f"Absolute Error ({args.metric_type})")
    # plt.title("Reasoning Length vs. Error")
    #
    # plt.subplot(1, 2, 2)
    # plt.scatter(reasoning_lengths, llm_scores_list)
    # plt.xlabel("Reasoning Length (tokens)")
    # plt.ylabel(f"LLM Predicted Score ({args.metric_type})")
    # plt.title("Reasoning Length vs. LLM Score")
    # plt.tight_layout()
    # plt.show() # or save to file: plt.savefig("correlation_plots.png") 