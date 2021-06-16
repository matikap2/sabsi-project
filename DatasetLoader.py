import pandas as pd

"""!
@brief Dictonary used to colour input colour mapping
"""
COLOURS_MAP = {'Red': 0, 'Green': 1 , 'Blue': 2, 
              'Yellow': 3, 'Orange': 4, 'Pink': 5, 
              'Purple': 6, 'Brown': 7, 'Grey': 8, 
              'Black': 9, 'White': 10, 'Unknown': 11} 

class DatasetLoader:
    """!
    @brief Load daset to train neural network
    @param[in] dataset_path path to csv file with dataset
    """
    def __init__(self, dataset_path: str):
        self._dataset_path = dataset_path
        self._loaded_data = list()
        self._input_data = list()
        self._results = list()
        self.load_data() 

    """!
    @brief Get path to current datset.
    @return Path to current dataset in string format
    """
    def get_dataset_path(self) -> str:
        return self._dataset_path

    """!
    @brief Set path to new datset.
    @param[in] dataset_path to new dataset in string format
    """  
    def set_dataset_path(self, dataset_path: str):
        self._dataset_path = dataset_path
    
    """!
    @brief Load data from .csv file and split it into input data(first three column) and correct colour set (last column)
    """
    def load_data(self):
        print("Data loaded from: " + self._dataset_path)
        self._loaded_data =  pd.read_csv(self._dataset_path, delimiter=';')
        for data_row in self._loaded_data.values.tolist():
            self._input_data.append(data_row[0:3])
            self._results.append(COLOURS_MAP.get(data_row[3], COLOURS_MAP.get('Unknown')))

        return self._input_data, self._results

    """!
    @brief Print data used to train neural network
    """
    def print_loaded_data(self):
        if len(self._loaded_data) != 0 :
            print(self._loaded_data)
        else:
            print("Empty data buffer")