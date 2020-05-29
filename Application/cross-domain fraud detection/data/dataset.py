import numpy as np
import torch.utils.data


class Mydataset(torch.utils.data.Dataset):

    def __init__(self, path, ratio):
        """
            :param
            path: the path of dataset
            ratio: the ritio of oversample
        """
        self.path = path
        file = open(self.path, 'r')
        self.lines = file.readlines()
        file.close()
        self.categorical_columns = 50
        self.numerical_columns = 6
        self.maxnum_events = 11  # 10 historical event 1 target payment event
        self.nb_features_per_event = self.categorical_columns + self.numerical_columns

        length = len(self.lines)
        self.weight = []
        black_num = 0
        for i in range(length):
            tmp = self.lines[i].split(' ')
            if int(tmp[2]) == 1:
                self.weight.append(ratio)
                black_num += 1
            else:
                self.weight.append(1)
        print(path, 'black num:', black_num, 'white num:', length - black_num)

    def __len__(self):
        return len(self.lines)

    def get_weight(self):
        return self.weight

    def __getitem__(self, index):
        """
            :param: index
            return  (ids, values, seq_length, label, seq_mask)
            ids: the ids of fileds
            values: values of fields
            seq_length: the length of historical events
            label: the label of the target payment
            seq_mask: the attention mask
        """
        line = self.lines[index]
        line = line.strip()
        line = line.split(' ')
        event_id = line.pop(0)
        num_events = int(line.pop(0))
        label = np.array(int(line.pop(0)))
        valid_features = self.nb_features_per_event * self.maxnum_events

        line = line[-valid_features:]
        ids, values = zip(*[x.split(':') for x in line])
        ids = list(ids)
        values = list(values)
        # the id, values of the target payment
        driver_ids = np.array(ids[-self.nb_features_per_event:]).reshape((1, -1))
        driver_values = np.array(values[-self.nb_features_per_event:]).reshape((1, -1))
        # the id, values of the historical events
        detail_ids = np.array(ids[:-self.nb_features_per_event]).reshape((-1, self.nb_features_per_event))
        detail_values = np.array(values[:-self.nb_features_per_event]).reshape((-1, self.nb_features_per_event))

        seq_length = self.maxnum_events - 1

        if num_events < self.maxnum_events - 1:
            seq_length = num_events
            detail_ids = np.concatenate(
                (detail_ids, np.zeros((self.maxnum_events - 1 - seq_length, self.nb_features_per_event))), axis=0)
            detail_values = np.concatenate(
                (detail_values, np.zeros((self.maxnum_events - 1 - seq_length, self.nb_features_per_event))), axis=0)

        if seq_length <= 0:
            seq_length = 1
        seq_length = np.array(seq_length)
        seq_mask = np.zeros(10, dtype='float32')
        seq_mask[seq_length:] = 1
        ids = np.concatenate((detail_ids, driver_ids), axis=0).astype(float).astype('int64')
        values = np.concatenate((detail_values, driver_values), axis=0).astype('float32')
        return ids, values, seq_length, label, seq_mask
