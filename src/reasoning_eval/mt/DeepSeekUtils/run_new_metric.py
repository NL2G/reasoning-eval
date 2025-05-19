from math import e
from mt_metrics_eval import meta_info, data, tasks
from tqdm import tqdm
import os
from DeepSeekR1Score import DeepSeekR1Score
import pandas as pd


LANG_DIR = {
    "en": "English",
    "de": "German",
    "fr": "French",
    "es": "Spanish",
    "he": "Hebrew",
    "zh": "Chinese",
}

class MetricEvaluator:
    def __init__(self, metric_name: str, language_pairs: list[str], dataset: str = 'wmt23',
                 evs_dict: dict[tuple, data.EvalSet] | None = None):
        """
        Initialize the evaluator with a metric name, language pairs, and evaluation sets.
        If evs_dict is not provided, it will be created using data.EvalSet.
        """
        self.metric_name = metric_name
        self.language_pairs = language_pairs
        self.dataset = dataset
        if dataset == 'wmt23':
            self.task_method = tasks.WMT23
        elif dataset == 'wmt24':
            self.task_method = tasks.WMT24
        else:
            raise ValueError(f"Unknown dataset: {dataset}. Please define it in the init method.")
        if evs_dict is None:
            print("Loading evaluation sets...")
            self.evs_dict = {(dataset, lp): data.EvalSet(dataset, lp, True)
                             for lp in tqdm(language_pairs, desc="Loading EvalSets")}
        else:
            self.evs_dict = evs_dict
        
    def compute_and_add_metrics(self, metric, ref_free=True):
        """Compute metric scores and add them to each EvalSet."""
        print("Computing and adding metric scores...")
        all_run_data = []
        
        for lp in tqdm(self.language_pairs, desc="Processing language pairs"):
            evs = self.evs_dict[(self.dataset, lp)]
            src_lang, tgt_lang = lp.split('-')
            
            # Try to get human/golden MQM scores if available
            golden_mqm_scores = {}
            for sys_name in evs.sys_names:
                if 'mqm' in evs.human_score_names:
                    try:
                        # Correct order: first parameter is level ('seg'), second is scorer ('mqm')
                        scores = evs.Scores('seg', 'mqm')
                        if scores and sys_name in scores:
                            golden_mqm_scores[sys_name] = scores[sys_name]
                    except Exception as e:
                        print(f"Warning: Could not get MQM scores for {sys_name}: {e}")
            
            seg_scores = None
            sys_scores = None
            print("ref-free: ", ref_free)
            for refname, ref in tqdm(evs.all_refs.items(), desc=f"Processing refs for {lp}", leave=False):
                print(f"Processing {lp} - {refname}...")
                output_file = f"outputs/{lp}_{refname}_{self.metric_name}.json"
                
                if (seg_scores is None or sys_scores is None) or not ref_free:
                    seg_scores, sys_scores, full_answers = metric(evs.src, ref, evs.sys_outputs, output_file, LANG_DIR[src_lang], LANG_DIR[tgt_lang])
                
                # Read the generated CSV file and add language pair and golden MQM if available
                csv_file = output_file.replace('json', 'csv')
                if os.path.exists(csv_file):
                    df = pd.read_csv(csv_file)
                    df['language_pair'] = lp
                    
                    # Add golden MQM score for each segment
                    for sys_name in df['system'].unique():
                        if sys_name in golden_mqm_scores:
                            # Add golden MQM score for each segment
                            for i, idx in enumerate(df[df['system'] == sys_name].index):
                                if i < len(golden_mqm_scores[sys_name]):
                                    df.loc[idx, 'golden_mqm_score'] = golden_mqm_scores[sys_name][i]
                                else:
                                    df.loc[idx, 'golden_mqm_score'] = None
                    
                    # Handle column renaming for consistency
                    if 'score' in df.columns and 'llm_mqm_score' not in df.columns:
                        df['llm_mqm_score'] = df['score']
                    
                    # Keep only essential columns in a clean format
                    essential_columns = ['system', 'src_segment', 'hyp_segment', 'language_pair', 
                                        'golden_mqm_score', 'llm_mqm_score', 'reasoning_tokens', 'reasoning_content']
                    
                    # Keep only columns that exist in the dataframe
                    existing_columns = [col for col in essential_columns if col in df.columns]
                    df = df[existing_columns]
                    
                    # Save updated CSV
                    df.to_csv(csv_file, index=False)
                    all_run_data.append(df)
                
                evs.AddMetric(self.metric_name, {refname}, 'sys', sys_scores, replace=False)
                evs.AddMetric(self.metric_name, {refname}, 'seg', seg_scores, replace=False)
        
        # Combine all data from this run into a single summary CSV
        if all_run_data:
            combined_df = pd.concat(all_run_data, ignore_index=True)
            combined_df.to_csv(f"outputs/run_summary_{self.metric_name}.csv", index=False)
                
        print("Metric scores added.\n")

    def set_primary_metrics(self):
        """Set the newly computed metric as primary for evaluation."""
        print("Setting primary metrics...")
        for evs in tqdm(self.evs_dict.values(), desc="Setting primary metrics"):
            evs.SetPrimaryMetrics(evs.primary_metrics | {self.metric_name})
        print("Primary metrics set.\n")

    def run_initial_evaluation(self):
        """
        Run evaluation with the current metrics without significance testing.
        Returns the new results and task weights.
        """
        print("Running initial evaluation (no significance testing)...")
        wmt_tasks, wts = self.task_method(self.language_pairs, k=0)
        new_results = wmt_tasks.Run(eval_set_dict=self.evs_dict)
        print("Initial evaluation complete.\n")
        return new_results, wts

    def print_results_table(self, new_results, wts):
        """Print the results table for the current evaluation."""
        avg_corrs = new_results.AverageCorrs(wts)
        table = new_results.Table(
            metrics=list(avg_corrs),
            initial_column=avg_corrs,
            initial_column_header='avg-corr',
            attr_list=['lang', 'level', 'corr_fcn'],
            nicknames={'KendallWithTiesOpt': 'acc-t'},
            fmt='text',
            baselines_metainfo=meta_info.WMT23
        )
        print("Evaluation Results:")
        print(table)
        return avg_corrs

    def set_primary_metrics_for_significance(self):
        """Set primary metrics for significance testing."""
        print("Setting primary metrics for significance testing...")
        for evs in tqdm(self.evs_dict.values(), desc="Setting significance metrics"):
            evs.SetPrimaryMetrics({'Random-sysname', self.metric_name, 'eBLEU'})
        print("Significance primary metrics set.\n")

    def run_significance_comparison(self, k: int = 50):
        """
        Run evaluation with significance testing.
        Returns the new results and task weights.
        """
        print("Running significance testing...")
        self.set_primary_metrics_for_significance()
        wmt23_tasks, wts = tasks.WMT23(self.language_pairs, k=k)
        new_results = wmt23_tasks.Run(eval_set_dict=self.evs_dict)
        print("Significance testing complete.\n")
        return new_results, wts

    def print_significance_table(self, new_results):
        """
        Print significance testing results including the p-value matrix.
        """
        print("Generating significance results...")
        _, main_task_weights = tasks.WMT23(k=0)
        avg_corrs, matrix = new_results.AverageCorrMatrix(main_task_weights)
        table = new_results.Table(
            metrics=list(avg_corrs),
            initial_column=avg_corrs,
            initial_column_header='avg-corr',
            attr_list=['lang', 'level', 'corr_fcn'],
            nicknames={'KendallWithTiesOpt': 'acc-t'},
            fmt='text',
            baselines_metainfo=meta_info.WMT23
        )
        print("Significance Results:")
        print(table)
        print("\nP-value Matrix:")
        print(tasks.MatrixString(avg_corrs, matrix, probs=True))
        print()


def main():

    # Define language pairs and load evaluation sets
    lps = ['en-de', 'he-en', 'zh-en']
    #lps = ['en-de']
    metric_name = 'deepseek-r1-llama8b'
    model = 'deepseek-ai/DeepSeek-R1-Distill-Llama-8B'
    dsr1 = DeepSeekR1Score(models=[model], arguments={'include_reasoning': False})
    evaluator = MetricEvaluator(f"{metric_name}_{model.split('/')[-1]}", lps)

    # Compute and add metric scores, then set them as primary
    evaluator.compute_and_add_metrics(dsr1)
    evaluator.set_primary_metrics()

    # Run initial evaluation and print results
    new_results, wts = evaluator.run_initial_evaluation()
    evaluator.print_results_table(new_results, wts)

    # Run significance testing and print significance results
    new_results, wts = evaluator.run_significance_comparison(k=50)
    evaluator.print_significance_table(new_results)


if __name__ == '__main__':
    '''
    import glob
    import pandas as pd
    paths = glob.glob("outputs/*.csv")
    for path in paths:
        df = pd.read_csv(path)
        assert all([len(l)>5 for l in list(df['response'])]), path
    raise ValueError
    '''
    main()
