import os
import pandas as pd
from Utils.utils_inference import InferenceUtils
import Utils.utils_emphysema
import Utils.utils_body_composition
from Utils.statistics import get_rmse_w_ci, get_rmse
from Utils.plot import bland_altman_plot
import numpy as np
from math import log10, sqrt
import nibabel as nib
from joblib import Parallel, delayed
from tqdm import tqdm
from skimage.metrics import mean_squared_error, peak_signal_noise_ratio, structural_similarity
import matplotlib.pyplot as plt


class GenerateAssessmentForRawData:
    def __init__(
            self,
            data_root='/path/to/data',
            in_data_dir='/lpath/to/data/directory'
    ):
        self.data_root = data_root
        self.data_index_dir = os.path.join(self.data_root, 'select_cases')
        self.data_output_dir = os.path.join(self.data_root, 'data.application')
        os.makedirs(self.data_output_dir, exist_ok=True)
        self.kernel_pair_type_list = [
            'B30f_B50f',
            'C_D',
            'FC10_FC51',
            'STANDARD_BONE',
            'STANDARD_LUNG'
        ]
        self.in_data_dir = in_data_dir

    def generate_data(self):
        """
        Prepare data / folder structure for the applications
        :return:
        """
        for kernel_pair_type in self.kernel_pair_type_list:
            in_full_record_csv = os.path.join(self.data_index_dir, f'{kernel_pair_type}.w_additional.csv')
            in_full_record_df = pd.read_csv(in_full_record_csv)
            kernel_pair_test_df = in_full_record_df.loc[
                (in_full_record_df['split'] == 'valid') |
                (in_full_record_df['split'] == 'test')]

            kernel_pair_type_out_dir = os.path.join(self.data_output_dir, kernel_pair_type)
            os.makedirs(kernel_pair_type_out_dir, exist_ok=True)
            index_record_list = []
            for record_index, kernel_pair_record in kernel_pair_test_df.iterrows():
                pid = kernel_pair_record['pid']
                pid_nii_file_name = f'{pid}.nii.gz'
                index_record_list.append({
                    'pid': pid,
                    'file_name': pid_nii_file_name
                })

                for domain in ['soft', 'hard']:
                    series_uid = kernel_pair_record[f'{domain}_uid']
                    in_nii = os.path.join(self.in_data_dir, f'{series_uid}.nii.gz')
                    out_nii_dir = os.path.join(kernel_pair_type_out_dir, domain, 'ct')
                    os.makedirs(out_nii_dir, exist_ok=True)
                    out_nii = os.path.join(out_nii_dir, f'{pid}.nii.gz')
                    ln_cmd = f'ln -sf {in_nii} {out_nii}'
                    os.system(ln_cmd)
            index_record_df = pd.DataFrame(index_record_list)
            index_record_csv = os.path.join(kernel_pair_type_out_dir, 'index.csv')
            print(f'Save index to {index_record_csv}')
            index_record_df.to_csv(index_record_csv, index=False)

    def run_emphysema_analysis(self):
        print(f'Run emphysema analysis')
        for kernel_pair_type in self.kernel_pair_type_list:
            for domain in ['hard', 'soft']:
                print(f'Process {kernel_pair_type} - {domain}')
                in_ct_dir = os.path.join(self.data_output_dir, kernel_pair_type, domain, 'ct')
                emphysema_dir = os.path.join(self.data_output_dir, kernel_pair_type, domain, 'emphysema')
                os.makedirs(emphysema_dir, exist_ok=True)
                emph_analyzer = Utils.utils_emphysema.EmphysemaAnalysis(in_ct_dir, emphysema_dir)
                emph_analyzer.generate_lung_mask()
                emph_analyzer.get_emphysema_mask()
                emph_analyzer.get_emphysema_measurement()

    def run_body_composition_analysis(self):
        print(f'Run body composition analysis')
        for kernel_pair_type in self.kernel_pair_type_list:
            for domain in ['hard', 'soft']:
                print(f'Process {kernel_pair_type} - {domain}')
                in_ct_dir = os.path.join(self.data_output_dir, kernel_pair_type, domain, 'ct')
                body_composition_dir = os.path.join(self.data_output_dir, kernel_pair_type, domain, 'body_composition')
                os.makedirs(body_composition_dir, exist_ok=True)
                bcomp_analyzer = Utils.utils_body_composition.BodyCompositionAnalyzer(
                    in_ct_dir,
                    body_composition_dir)
                bcomp_analyzer.generate_input_data()
                bcomp_analyzer.generate_run_sh()
                bcomp_analyzer.run_sh()


class ConversionPerformanceAnalyzer:
    def __init__(self, config, project_root, in_data_root, use_checkpoint):
        self.config = config
        self.project_root = project_root
        os.makedirs(self.project_root, exist_ok=True)
        self.in_data_root = in_data_root
        self.use_checkpoint = use_checkpoint

    def generate_data(self, scan_record_csv, in_data_dir):
        """
        Use both valid and test cohort as test cohort, since we don't have a validation set for pix2pix.
        :return:
        """
        # scan_record_csv = '/local_storage/Projects/KernelNormalization/select_cases/B30f_B50f.csv'
        print(f'Load {scan_record_csv}')
        scan_record_df = pd.read_csv(scan_record_csv)

        test_set_df = scan_record_df.loc[
            (scan_record_df['split'] == 'valid') |
            (scan_record_df['split'] == 'test')]

        # in_data_dir = '/local_storage/Data/NLST/NIfTI/T0_all'

        in_kernel = self.config['data']['in_kernel']
        test_scan_dir = os.path.join(self.project_root, 'test_scan')
        os.makedirs(test_scan_dir, exist_ok=True)
        print(f'Save test scan to {test_scan_dir}')

        for index, record in test_set_df.iterrows():
            pid = record['pid']
            series_uid = record[f'{in_kernel}_uid']
            in_nii = os.path.join(in_data_dir, str(series_uid) + '.nii.gz')
            out_nii = os.path.join(test_scan_dir, str(pid) + '.nii.gz')
            ln_cmd = f'ln -sf {in_nii} {out_nii}'
            os.system(ln_cmd)

        out_kernel = self.config['data']['out_kernel']
        test_scan_target_dir = os.path.join(self.project_root, 'test_scan_target')
        os.makedirs(test_scan_target_dir, exist_ok=True)
        print(f'Save target test scan to {test_scan_target_dir}')
        for index, record in test_set_df.iterrows():
            pid = record['pid']
            series_uid = record[f'{out_kernel}_uid']
            in_nii = os.path.join(in_data_dir, str(series_uid) + '.nii.gz')
            out_nii = os.path.join(test_scan_target_dir, str(pid) + '.nii.gz')
            ln_cmd = f'ln -sf {in_nii} {out_nii}'
            os.system(ln_cmd)

    def generate_data_v2(self):
        source_kernel = self.config['data']['in_kernel']
        target_kernel = self.config['data']['out_kernel']

        test_scan_dir = os.path.join(self.project_root, 'test_scan')
        if os.path.exists(test_scan_dir):
            rm_cmd = f'rm -rf {test_scan_dir}'
            os.system(rm_cmd)
        source_ct_dir = os.path.join(self.in_data_root, source_kernel, 'ct')
        ln_cmd = f'ln -sf {source_ct_dir} {test_scan_dir}'
        os.system(ln_cmd)

        test_scan_target_dir = os.path.join(self.project_root, 'test_scan_target')
        if os.path.exists(test_scan_target_dir):
            rm_cmd = f'rm -rf {test_scan_target_dir}'
            os.system(rm_cmd)
        target_ct_dir = os.path.join(self.in_data_root, target_kernel, 'ct')
        ln_cmd = f'ln -sf {target_ct_dir} {test_scan_target_dir}'
        os.system(ln_cmd)

    def run_kernel_conversion(self):
        infer_utils = InferenceUtils(self.config, self.use_checkpoint)
        infer_utils.convert_nii_dir(
            in_nii_dir=os.path.join(self.project_root, 'test_scan'),
            out_nii_dir=os.path.join(self.project_root, 'test_scan_converted')
        )
        InferenceUtils.generate_fov_mask(
            in_nii_dir=os.path.join(self.project_root, 'test_scan'),
            out_ppr_mask_dir=os.path.join(self.project_root, 'test_scan.ppr')
        )
        InferenceUtils.correct_non_fov_region(
            in_nii_dir=os.path.join(self.project_root, 'test_scan_converted'),
            ppr_mask_dir=os.path.join(self.project_root, 'test_scan.ppr'),
            out_nii_dir=os.path.join(self.project_root, 'test_scan_converted.ppr_corrected')
        )

    @staticmethod
    def PSNR(original, compressed):
        mse = np.mean((original - compressed) ** 2)
        if (mse == 0):  # MSE is zero means no noise is present in the signal .
            # Therefore PSNR have no importance.
            return 100
        max_pixel = 255.0
        psnr = 20 * log10(max_pixel / sqrt(mse))
        return psnr

    def run_intensity_based_metrics(self):
        index_csv = os.path.join(self.in_data_root, 'index.csv')
        index_df = pd.read_csv(index_csv)

        def process_single_record(record_dict):
            predict_raw_nii = record_dict['predict_raw']
            predict_convert_nii = record_dict['predict_convert']
            target_nii = record_dict['target']
            predict_raw_data = nib.load(predict_raw_nii).get_fdata()
            predict_convert_data = nib.load(predict_convert_nii).get_fdata()
            target_data = nib.load(target_nii).get_fdata() 

            lung_range = [-1000, 0] #Lung window for PSNR and SSIM
            muscle_range = [-29, 150] #From paper
            SAT_range = [-190, -30] # From paper 
            musc_SAT = [-190, 190]

            # Used for entire HU range of [-1024, 3072]
            # predict_raw_data = np.clip(
            #     predict_raw_data, self.config['data']['clip_range'][0], self.config['data']['clip_range'][1])
            # predict_convert_data = np.clip(
            #     predict_convert_data, self.config['data']['clip_range'][0], self.config['data']['clip_range'][1])
            # target_data = np.clip(
            #     target_data, self.config['data']['clip_range'][0], self.config['data']['clip_range'][1]) 
            
            predict_raw_data = np.clip(
                predict_raw_data, SAT_range[0], SAT_range[1])
            predict_convert_data = np.clip(
                predict_convert_data, SAT_range[0], SAT_range[1])
            target_data = np.clip(
                target_data, SAT_range[0], SAT_range[1])

            result_dict = {}
            for metric_tag, metric_method in zip(
                ['mse'],
                [mean_squared_error]
            ):
                result_dict[metric_tag] = {
                    'pid': record_dict['pid'],
                    'raw': metric_method(target_data, predict_raw_data),
                    'convert': metric_method(target_data, predict_convert_data)
                }

            for metric_tag, metric_method in zip(
                ['psnr', 'ssim'],
                [peak_signal_noise_ratio, structural_similarity]
            ):
                data_range = SAT_range[1] - SAT_range[0]
                result_dict[metric_tag] = {
                    'pid': record_dict['pid'],
                    'raw': metric_method(target_data, predict_raw_data, data_range=data_range),
                    'convert': metric_method(target_data, predict_convert_data, data_range=data_range)
                }

            return result_dict

        process_record_list = []
        for _, scan_record in index_df.iterrows():
            nii_file_name = scan_record['file_name']
            process_record_list.append({
                'pid': scan_record['pid'],
                'predict_raw': os.path.join(self.project_root, 'test_scan', nii_file_name),
                'predict_convert': os.path.join(self.project_root, 'test_scan_converted.ppr_corrected', nii_file_name),
                'target': os.path.join(self.project_root, 'test_scan_target', nii_file_name)
            })

        result_list = Parallel(
            n_jobs=4,
            prefer='threads'
        )(delayed(process_single_record)(record_dict)
          for record_dict in tqdm(process_record_list,
                                  total=len(process_record_list),
                                  desc='Get image-based metrics'))

        # result_list = [result_item for sublist in result_list for result_item in sublist]
        for metric_tag in ['mse', 'psnr', 'ssim']:
            metric_data_list = [result_dict[metric_tag] for result_dict in result_list]

            metric_df = pd.DataFrame(metric_data_list)
            metric_csv = os.path.join(self.project_root, f'metric_SAT_{metric_tag}.csv')
            print(f'Save to {metric_csv}')
            metric_df.to_csv(metric_csv, index=False)

            # Also report the mean and std, before and after the conversion
            summary_data_dict = {}
            for phase in ['raw', 'convert']:
                metric_mean = np.nanmean(metric_df[phase].to_list())
                metric_std = np.nanstd(metric_df[phase].to_list())
                summary_data_dict[phase] = {
                    'mean': metric_mean,
                    'std': metric_std}
            summary_txt = os.path.join(self.project_root, f'metric_SAT_{metric_tag}.txt')
            with open(summary_txt, 'w') as file:
                file.write('Raw: {mean:.4f} ({std:.4f})\n'.format(
                    mean=summary_data_dict['raw']['mean'],
                    std=summary_data_dict['raw']['std']))
                file.write('Converted: {mean:.4f} ({std:.4f})\n'.format(
                    mean=summary_data_dict['convert']['mean'],
                    std=summary_data_dict['convert']['std']))

    def run_emphysema_analysis(self):
        source_kernel = self.config['data']['in_kernel']
        target_kernel = self.config['data']['out_kernel']

        group_list = ['test_scan_converted']
        emphysema_experiment_root = os.path.join(self.project_root, 'emphysema')
        os.makedirs(emphysema_experiment_root, exist_ok=True)
        for group in group_list:
            print(f'Process {group}')
            in_ct_dir = os.path.join(self.project_root, group)
            project_dir = os.path.join(emphysema_experiment_root, group)
            os.makedirs(project_dir, exist_ok=True)

            emph_analyzer = Utils.utils_emphysema.EmphysemaAnalysis(in_ct_dir, project_dir)
            emph_analyzer.generate_lung_mask()
            emph_analyzer.get_emphysema_mask()
            emph_analyzer.get_emphysema_measurement()

        # Soft link the already processed application data
        for group, kernel in zip(['test_scan', 'test_scan_target'], [source_kernel, target_kernel]):
            in_project_dir = os.path.join(self.in_data_root, kernel, 'emphysema')
            out_project_dir = os.path.join(emphysema_experiment_root, group)
            if os.path.exists(out_project_dir):
                rm_cmd = f'rm -rf {out_project_dir}'
                os.system(rm_cmd)
            ln_cmd = f'ln -sf {in_project_dir} {out_project_dir}'
            os.system(ln_cmd)

        # Get the RMSE (95%CI) and Bland-Altman plot.

        emph_score_dict = {}
        group_list = ['test_scan', 'test_scan_converted', 'test_scan_target']
        for group in group_list:
            group_emph_df = pd.read_csv(os.path.join(self.project_root, 'emphysema', group, 'emph.csv'))
            for index, record in group_emph_df.iterrows():
                pid = record['pid']
                emph_score = record['emph_score']

                if pid not in emph_score_dict:
                    emph_score_dict[pid] = {}
                emph_score_dict[pid][group] = emph_score

        emph_score_record_list = []
        for pid, score_dict in emph_score_dict.items():
            score_dict['pid'] = pid
            emph_score_record_list.append(score_dict)

        emph_score_df = pd.DataFrame(emph_score_record_list)
        emph_score_csv = os.path.join(self.project_root, 'emphysema', 'emph_score.csv')
        print(f'Save to {emph_score_csv}')
        emph_score_df.to_csv(emph_score_csv, index=False)

        in_to_target_rmse, in_to_target_rmse_ci = get_rmse_w_ci(
            emph_score_df['test_scan'].to_list(), emph_score_df['test_scan_target'].to_list())
        converted_to_target_rmse, converted_to_target_rmse_ci = get_rmse_w_ci(
            emph_score_df['test_scan_converted'].to_list(), emph_score_df['test_scan_target'].to_list())

        rmse_txt = os.path.join(self.project_root, 'emphysema', 'emph_rmse.txt')
        with open(rmse_txt, 'w') as file:
            file.write(f'Raw RMSE: {in_to_target_rmse} '
                       f'[{in_to_target_rmse_ci[0]}, {in_to_target_rmse_ci[1]}]\n')
            file.write(f'Converted RMSE: {converted_to_target_rmse} '
                       f'[{converted_to_target_rmse_ci[0]}, {converted_to_target_rmse_ci[1]}]\n')

        bland_altman_plot(
            gt_list=emph_score_df['test_scan_target'].to_list(),
            pred_list=emph_score_df['test_scan'].to_list(),
            gt_label='Target domain',
            pred_label='Source domain',
            x_plot_range=[0, 40],
            y_plot_range=[-40, 40],
            out_png=os.path.join(self.project_root, 'emphysema', 'bland_altman.raw.png')
        )

        bland_altman_plot(
            gt_list=emph_score_df['test_scan_target'].to_list(),
            pred_list=emph_score_df['test_scan_converted'].to_list(),
            gt_label='Target domain',
            pred_label='Target domain (converted)',
            x_plot_range=[0, 40],
            y_plot_range=[-40, 40],
            out_png=os.path.join(self.project_root, 'emphysema', 'bland_altman.converted.png')
        )

    def run_body_composition_analysis(self, platform_tag):
        source_kernel = self.config['data']['in_kernel']
        target_kernel = self.config['data']['out_kernel']

        # group_list = ['test_scan', 'test_scan_converted', 'test_scan_target']
        # group_list = ['test_scan', 'test_scan_converted.ppr_corrected', 'test_scan_target']
        group_list = ['test_scan_converted.ppr_corrected']
        component_list = ['Muscle', 'SAT'] 
       

        bcomp_project_dir = os.path.join(self.project_root, 'body_composition')
        os.makedirs(bcomp_project_dir, exist_ok=True)
        for group in group_list:
            print(f'Process {group}')
            in_ct_dir = os.path.join(self.project_root, group)
            project_dir = os.path.join(bcomp_project_dir, group)
            os.makedirs(project_dir, exist_ok=True)

            bcomp_analyzer = Utils.utils_body_composition.BodyCompositionAnalyzer(
                in_ct_dir,
                project_dir)
            bcomp_analyzer.generate_input_data()
            bcomp_analyzer.generate_run_sh()
            bcomp_analyzer.run_sh()

        # Soft link the already processed application data
        for group, kernel in zip(['test_scan', 'test_scan_target'], [source_kernel, target_kernel]):
            in_project_dir = os.path.join(self.in_data_root, kernel, 'body_composition')
            out_project_dir = os.path.join(bcomp_project_dir, group)
            if os.path.exists(out_project_dir):
                rm_cmd = f'rm -rf {out_project_dir}'
                os.system(rm_cmd)
            ln_cmd = f'ln -sf {in_project_dir} {out_project_dir}'
            os.system(ln_cmd)

        # Gather the measurements: muscle / SAT  #for every group, looping through the patients and getting the respective measurements. The code here checks for the output from the old pipeline.
        # It has to be updated for the new pipeline since the outputs from the old pipeline and new pipeline do not match.
        # New parameters are Muscle_area and SAT_area. Index into these two headers to obtain the parameters.
        # This old code uses the long format based on the output of the old pipeline. Now, the pipeline outputs a wide format so the code has to be modified.
        area_dict = {
            'Muscle': {},
            'SAT': {}
        }
        group_list = ['test_scan', 'test_scan_converted.ppr_corrected', 'test_scan_target']
        for group in group_list:
            measure_df = pd.read_csv(os.path.join(
                self.project_root, 'body_composition', group, 'Output', 'measurement.csv'))
            for nii_file_name, subject_df in measure_df.groupby(by='filename'):
                #print(subject_df)
                pid = nii_file_name.replace('.nii.gz', '')
            #    if len(subject_df.index) != 3:
            #        continue
                for component in component_list:
                    measure_list = subject_df[f'{component}_area'].iloc[0] #Must change this line as well.
            #        sum_area = np.sum(measure_list)
                    if pid not in area_dict[component]:
                        area_dict[component][pid] = {}
                    area_dict[component][pid][group] = measure_list

        area_record_list_dict = {}
        for component in component_list:
            component_area_dict = area_dict[component]
            area_record_list = []
            for pid, subject_area_dict in component_area_dict.items():
                subject_area_dict['pid'] = pid
                area_record_list.append(subject_area_dict)
            area_record_list_dict[component] = area_record_list

        for component in component_list:
            component_df = pd.DataFrame(area_record_list_dict[component])
            component_csv = os.path.join(self.project_root, 'body_composition', f'{component}.csv')
            print(f'Save to {component_csv}')
            component_df.to_csv(component_csv, index=False)

        # Get the Bland-Altman plot
        x_range_list = [
            [200, 600],  # Muscle
            [0, 1500]  # SAT
        ]
        y_range_list = [
            [-200, 200], #original is [-100,100]
            [-200, 200]
        ]
        for component, x_range, y_range in zip(component_list, x_range_list, y_range_list):
            component_csv = os.path.join(self.project_root, 'body_composition', f'{component}.csv')
            print(f'Load {component_csv}')
            component_df = pd.read_csv(component_csv)

            in_to_target_rmse, in_to_target_rmse_ci = get_rmse_w_ci(
                component_df['test_scan'].to_list(), component_df['test_scan_target'].to_list())
            # converted_to_target_rmse, converted_to_target_rmse_ci = get_rmse_w_ci(
            #     component_df['test_scan_converted'].to_list(), component_df['test_scan_target'].to_list())
            converted_to_target_rmse, converted_to_target_rmse_ci = get_rmse_w_ci(
                component_df['test_scan_converted.ppr_corrected'].to_list(), component_df['test_scan_target'].to_list())


            rmse_txt = os.path.join(self.project_root, 'body_composition', f'{component}_rmse.txt')
            with open(rmse_txt, 'w') as file:
                file.write(f'Raw RMSE: {in_to_target_rmse} '
                           f'[{in_to_target_rmse_ci[0]}, {in_to_target_rmse_ci[1]}]\n')
                file.write(f'Converted RMSE: {converted_to_target_rmse} '
                           f'[{converted_to_target_rmse_ci[0]}, {converted_to_target_rmse_ci[1]}]\n')
            bland_altman_plot(
                gt_list=component_df['test_scan_target'].to_list(),
                pred_list=component_df['test_scan'].to_list(),
                gt_label='Target domain',
                pred_label='Source domain',
                x_plot_range=x_range,
                y_plot_range=y_range,
                out_png=os.path.join(self.project_root, 'body_composition', f'{component}.bland_altman.raw.png')
            )
            bland_altman_plot(
                gt_list=component_df['test_scan_target'].to_list(),
                # pred_list=component_df['test_scan_converted'].to_list(),
                pred_list=component_df['test_scan_converted.ppr_corrected'].to_list(),
                gt_label='Target domain',
                pred_label='Target domain (converted)',
                x_plot_range=x_range,
                y_plot_range=y_range,
                out_png=os.path.join(self.project_root, 'body_composition', f'{component}.bland_altman.converted.png')
            )


class EpochPerformanceAnalyzer:
    def __init__(self, config, train_root, project_root, in_data_root, epoch_range):
        self.config = config
        self.train_root = train_root
        self.project_root = project_root  # The root location of all output files.
        os.makedirs(self.project_root, exist_ok=True)
        self.in_data_root = in_data_root
        self.epoch_range = epoch_range
        self.analyzer_list = self._get_analyzer_list()

    def _get_analyzer_list(self):
        analyzer_list = []
        for epoch_index in self.epoch_range:
            analyze_project_dir = os.path.join(self.project_root, f'checkpoint_{epoch_index}')
            os.makedirs(analyze_project_dir, exist_ok=True)
            use_checkpoint = os.path.join(self.train_root, f'checkpoint_{epoch_index}.tar')
            analyzer = ConversionPerformanceAnalyzer(
                self.config,
                analyze_project_dir,
                self.in_data_root,
                use_checkpoint)
            analyzer_list.append(analyzer)

        return analyzer_list

    def run_kernel_conversion(self):
        for analyzer in self.analyzer_list:
            analyzer.generate_data_v2()
            analyzer.run_kernel_conversion()

    def run_intensity_based_metrics(self):
        for analyzer in self.analyzer_list:
            analyzer.run_intensity_based_metrics()

    def run_emphysema_analysis(self):
        for analyzer in self.analyzer_list:
            analyzer.run_emphysema_analysis()

    def run_body_composition_analysis(self):
        for analyzer in self.analyzer_list:
            analyzer.run_body_composition_analysis(platform_tag='masi')

    def get_epoch_trend_record_csv(self):
        field_tag_list = ['emph', 'bcomp', 'image_mse', 'image_psnr', 'image_ssim']
        record_list = []
        for epoch in self.epoch_range:
            record_dict = {'epoch': epoch}

            analyze_project_dir = os.path.join(self.project_root, f'checkpoint_{epoch}')

            if 'emph' in field_tag_list:
                # report rmse against the target domain
                emph_score_csv = os.path.join(analyze_project_dir, 'emphysema', 'emph_score.csv')
                emph_score_df = pd.read_csv(emph_score_csv)
                emph_list_converted = emph_score_df['test_scan_converted'].to_list()
                emph_list_target = emph_score_df['test_scan_target'].to_list()
                record_dict['emph_rmse'] = get_rmse(emph_list_target, emph_list_converted)

            if 'bcomp' in field_tag_list:
                # report rmse of muscle and sat separately
                area_dict = {
                    'Muscle': {},
                    'SAT': {}}
                group_list = ['test_scan_converted.ppr_corrected', 'test_scan_target']
                for group in group_list:
                    measure_df = pd.read_csv(os.path.join(
                        self.project_root, 'body_composition', group, 'Output', 'measurement.csv'))
                    for nii_file_name, subject_df in measure_df.groupby(by='filename'):
                        pid = nii_file_name.replace('.nii.gz', '')
                        if len(subject_df.index) != 3:
                            continue
                        for component in area_dict.keys():
                            measure_list = subject_df[f'{component}_cm2'].to_list()
                            sum_area = np.sum(measure_list)
                            if pid not in area_dict[component]:
                                area_dict[component][pid] = {}
                            area_dict[component][pid][group] = sum_area

                area_record_list_dict = {}
                for component in area_dict.keys():
                    component_area_dict = area_dict[component]
                    area_record_list = []
                    for pid, subject_area_dict in component_area_dict.items():
                        subject_area_dict['pid'] = pid
                        area_record_list.append(subject_area_dict)
                    area_record_list_dict[component] = area_record_list

                for component in area_dict.keys():
                    component_df = pd.DataFrame(area_record_list_dict[component])
                    record_dict[f'bcomp_{component}_rmse'] = get_rmse(
                        component_df['test_scan_converted.ppr_corrected'].to_list(),
                        component_df['test_scan_target'].to_list())

            if 'image_mse' in field_tag_list:
                # Report the mean image mse against the target
                mse_csv = os.path.join(analyze_project_dir, 'metric_mse.csv')
                mse_df = pd.read_csv(mse_csv)
                mse_list = mse_df['convert'].to_list()
                record_dict['image_mse_mean'] = np.mean(mse_list)

            if 'image_psnr' in field_tag_list:
                psnr_csv = os.path.join(analyze_project_dir, 'metric_psnr.csv')
                psnr_df = pd.read_csv(psnr_csv)
                psnr_list = psnr_df['convert'].to_list()
                record_dict['image_psnr_mean'] = np.mean(psnr_list)

            if 'image_ssim' in field_tag_list:
                ssim_csv = os.path.join(analyze_project_dir, 'metric_ssim.csv')
                ssim_df = pd.read_csv(ssim_csv)
                ssim_list = ssim_df['convert'].to_list()
                record_dict['image_ssim_mean'] = np.mean(ssim_list)

            record_list.append(record_dict)

        record_df = pd.DataFrame(record_list)
        record_csv = os.path.join(self.project_root, 'epoch_perf_record.csv')
        print(f'Save to {record_csv}')
        record_df.to_csv(record_csv, index=False)

    def get_epoch_trend_plot(self):
        record_csv = os.path.join(self.project_root, 'epoch_perf_record.csv')
        record_df = pd.read_csv(record_csv)

        pass