import os.path as osp


class BaseImageDataset(object):

    def parse_data(self, data):
        pids = set()
        cams = set()
        for _, pid, camid in data:
            pids.add(pid)
            cams.add(camid)
        return len(pids), len(cams)

    def show_summary(self, train, query, gallery):
        num_train_pids, num_train_cams = self.parse_data(train)
        num_query_pids, num_query_cams = self.parse_data(query)
        num_gallery_pids, num_gallery_cams = self.parse_data(gallery)

        print('=> Loaded {}'.format(self.__class__.__name__))
        print('  ----------------------------------------')
        print('  subset   | # ids | # images | # cameras')
        print('  ----------------------------------------')
        print('  train    | {:5d} | {:8d} | {:9d}'.format(num_train_pids, len(train), num_train_cams))
        print('  query    | {:5d} | {:8d} | {:9d}'.format(num_query_pids, len(query), num_query_cams))
        print('  gallery  | {:5d} | {:8d} | {:9d}'.format(num_gallery_pids, len(gallery), num_gallery_cams))
        print('  ----------------------------------------')

    def get_num_pids(self, data):
        return self.parse_data(data)[0]

    def check_before_run(self, required_files):
        if isinstance(required_files, str):
            required_files = [required_files]

        for fpath in required_files:
            if not osp.exists(fpath):
                raise RuntimeError('"{}" is not found'.format(fpath))
