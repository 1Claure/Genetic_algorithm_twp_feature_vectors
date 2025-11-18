##################################################
#  File creation from IM-tention' mat files              
#  
#  Diana C. Vertiz del Valle
##################################################
import numpy as np
from scipy import signal
from scipy.io import loadmat, savemat
from modules.preprocessing.filtering import preprocess_signal_im_tention
import glob

def getSubjectFiles(folder, file_type='mat'):
    """get all the mat files names in a folder
    Args:
        folder (string): path to folder
    Returns:
        data_list: list with strings of the mat files names
    """
    source_path = folder + '/' 
    data_list = []    
    for name in glob.glob(source_path + '**/*.'+ file_type, recursive=True):      
        data_list.append(name)
    return data_list

def create_mat_files(file_path, file_type='calibration', rehabilitation_limb=1, \
                    skip_samples=0.5, mi_samples=2.5, rest_samples=2.5, calib_number_of_trials=None,\
                    ther_number_of_trials = None, joined=True, scaled=False, mi_only=True, filter=True, filtfilt=False):
    """
    Function to create mat files from IM-tention's mat files of a group of subjects inside a folder
    Args:
        file_path (str): path to the folder where the mat files are located
        file_type (str): the type of file to create. It can be 'calibration' or 'therapy'
        rehabilitation_limb (int): the limb to rehabilitate. It can be 0 or 1. 
                                1 is the rehabilitated limb used in therapy stage
        skip_samples (int): the length in seconds to skip after and before the MI starts. default is 0.5 seconds
        mi_samples (int): the length in seconds after MI mark. default is 2.5 seconds
        rest_samples (int): the length in seconds before MI mark. default is 2.5 seconds
        calib_number_of_trials (int): the number of trials to extract from calibration stage. default is None.
                                    If None, all the trials are extracted.
        ther_number_of_trials (int): the number of trials to extract from therapy stage. default is None.
                                    If None, all the trials are extracted.
        joined (bool): if the rest and MI signals are joined in the same matrix.
                       valid for calibration stage and therapy stage. See the possible structures below         
        scaled (bool): if the signals are scaled with IM-tention's factor (22369.61866666667)
        mi_only (bool): if only MI signals are extracted (valid only for therapy stage). See the possible structures below
    Returns:
        subject_data: dict of dicts with the posible structures:
            For calibration stage:
            {'subject_1': {'rest': rest_matrix, 'mi': mi_matrix}} ---> file_type='calibration', joined = False 
            {'subject_1': {'mi_rest': matrix}                     ---> file_type='calibration', joined = True
            For therapy stage:
            {'subject_1': {'rest': rest_matrix, 'mi': mi_matrix, 'target': target_matrix}} ---> file_type='therapy', joined = False
            {'subject_1': {'mi': mi_matrix, 'target': target_matrix}} ---> file_type='therapy', mi_only = True
            {'subject_1': {'mi_rest': matrix, 'target': target_matrix}} ---> file_type='therapy', joined = True, mi_only = False
    """
    mv_value = 22369.61866666667 
    name_lists = getSubjectFiles(file_path)
    print(f"Number of subjects found: {len(name_lists)}")
    subject_data = {}
    
    calib_trials = 0
    max_calib_trials = 0
    max_ther_trials = 0
    therapy_trials = 0

    for n, name in enumerate(name_lists):
        mat_contents = loadmat(name, squeeze_me=True)
        fs = mat_contents['sample_rate']
        #rest trial interval samples
        r_samples = int(rest_samples * fs)
        #motor imagery samples
        m_samples  = int(mi_samples * fs)
        #samples to skip
        s_samples = int(skip_samples * fs)

        #load calibration signals and marks
        calib_signal = mat_contents['calib_signals'].T
        # print("calib_signal:", calib_signal.shape)
        calib_marks = mat_contents['calib_task_marks']
        # print("calib_marks:", len(calib_marks))
        #the rehabilitated limb is marked with 1
        calib_limbs = mat_contents['calib_task_limbs']

        #load therapy signals and marks
        therapy_signal = mat_contents['attempt_signals'].T
        therapy_marks  = mat_contents['attempt_marks']
        # print("therapy_marks:", len(therapy_marks))
        
        max_calib_trials = len(calib_marks)//2
        max_ther_trials = len(therapy_marks)

        if calib_number_of_trials is None:
            calib_trials = max_calib_trials
        else:
            calib_trials = calib_number_of_trials  

        if ther_number_of_trials is None:        
            therapy_trials = max_ther_trials
        else:
            therapy_trials = ther_number_of_trials      
        
        # print("max_calib_trials:", max_calib_trials)
        # print("calib_trials:", calib_trials)

        if calib_trials > max_calib_trials:
            raise ValueError(f"The number of trials to extract is greater than the number of trials in the subject {n+1}")
        if therapy_trials > max_ther_trials:
            raise ValueError(f"The number of trials to extract is greater than the number of trials in the subject {n+1}")

        therapy_result = mat_contents['attempt_results'][:therapy_trials]
        #mark is the same for all trials in therapy
        mark = therapy_marks[0]

        if filter:
            calib_signal = preprocess_signal_im_tention(calib_signal, fs, filtfilt)
            therapy_signal = preprocess_signal_im_tention(therapy_signal, fs, filtfilt)

        if file_type == 'calibration':    
            print(f"Number of MI calibration trials for subject{n+1}: {calib_trials}")
            #[Ns * Nc * Nt]
            rest_matrix = np.zeros((r_samples-s_samples, calib_signal.shape[0], calib_trials))
            mi_matrix = np.zeros((m_samples-s_samples, calib_signal.shape[0], calib_trials))  

            if rehabilitation_limb == 1:
                indexes = np.where(calib_limbs == 1)[0]
            else:
                indexes = np.where(calib_limbs == 0)[0]
            for i, indx in enumerate(indexes):
                if i == calib_trials:
                    break
                for j,channel in enumerate(calib_signal):
                    midp = calib_marks[indx]
                    rest_matrix[:,j,i] = channel[midp-r_samples:midp-s_samples]
                    mi_matrix[:,j,i] = channel[midp+s_samples:midp+m_samples]

        if file_type == 'therapy':  
            print(f"Number of therapy trials for subject{n+1}: {therapy_trials}")              
            #[Ns * Nc * Nt]
            rest_matrix = np.zeros((r_samples-s_samples, therapy_signal.shape[1], therapy_trials))
            mi_matrix = np.zeros((m_samples-s_samples, therapy_signal.shape[1], therapy_trials))  

            for i, intention in enumerate(therapy_signal):
                # print(intention.shape)
                if i == therapy_trials:
                    break
                for j,channel in enumerate(intention):        
                    rest_matrix[:,j,i] = channel[mark-r_samples:mark-s_samples]
                    mi_matrix[:,j,i] = channel[mark+s_samples:mark+m_samples]  
        if scaled:
            rest_matrix = rest_matrix/mv_value
            mi_matrix = mi_matrix/mv_value
        
        if not joined:
            subject_data['subject_' + str(n+1)] = {'rest': rest_matrix, 'mi': mi_matrix}
        else:
            subject_data['subject_' + str(n+1)] = {'mi_rest': np.concatenate((mi_matrix, rest_matrix), axis=2)}
       
        if mi_only and file_type == 'therapy':
            subject_data['subject_' + str(n+1)] = {'mi': mi_matrix}

        if file_type == 'therapy':
            subject_data['subject_' + str(n+1)]['target'] = therapy_result


    return subject_data    
        

if __name__ == '__main__':
    mat_contents = loadmat('./data/im_tention_signals/Acevedo/6-9_02-10-23.mat', squeeze_me=True)
    print(mat_contents['attempt_results'])

