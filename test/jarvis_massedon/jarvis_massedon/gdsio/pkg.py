"""
This module provides classes and methods to launch the LabstorIpcTest application.
LabstorIpcTest is ....
"""
from jarvis_cd.basic.pkg import Application
import re
from jarvis_util import *


class Gdsio(Application):
    """
    This class provides methods to launch the LabstorIpcTest application.
    """
    def _init(self):
        """
        Initialize paths
        """
        pass

    def _configure_menu(self):
        """
        Create a CLI menu for the configurator method.
        For thorough documentation of these parameters, view:
        https://github.com/scs-lab/jarvis-util/wiki/3.-Argument-Parsing

        :return: List(dict)
        """
        return [
            {
                'name': 'async',
                'msg': 'Wether to run the package as async',
                'type': bool,
                'default': False,
            },
            {
                'name': 'nodes',
                'msg': 'The number of nodes to launch on',
                'type': int,
                'default': 1,
            },
            {
                'name': 'file',
                'msg': 'File name to test',
                'type': str,
                'default': '',
            },
            {
                'name': 'directory',
                'msg': 'Directory name to test',
                'type': str,
                'default': '',
            },
            {
                'name': 'gpu_index',
                'msg': 'GPU index (refer nvidia-smi)',
                'type': int,
                'default': 0,
            },
            {
                'name': 'numa',
                'msg': 'NUMA node',
                'type': int,
                'default': None,
            },
            {
                'name': 'memtype',
                'msg': 'Memory type (0-cudaMalloc, 1-cuMem, 2-cudaMallocHost, 3-malloc, 4-mmap)',
                'type': int,
                'default': 0,
            },
            {
                'name': 'threads',
                'msg': 'Number of threads for a job',
                'type': int,
                'default': 1,
            },
            {
                'name': 'size',
                'msg': 'File size (K|M|G)',
                'type': str,
                'default': '1G',
            },
            {
                'name': 'offset',
                'msg': 'Start offset (K|M|G)',
                'type': str,
                'default': '0',
            },
            {
                'name': 'io_size',
                'msg': 'IO size (K|M|G) <min_size:max_size:step_size>',
                'type': str,
                'default': '1M',
            },
            {
                'name': 'xfer_type',
                'msg': ' [0(GPU_DIRECT), 1(CPU_ONLY), 2(CPU_GPU), 3(CPU_ASYNC_GPU), 4(CPU_CACHED_GPU), 5(GPU_DIRECT_ASYNC), 6(GPU_BATCH), 7(GPU_BATCH_STREAM)]',
                'type': int,
                'default': 0,
            },
            {
                'name': 'nvlinks',
                'msg': 'Enable nvlinks',
                'type': bool,
                'default': False,
            },
            {
                'name': 'skip_bufregister',
                'msg': 'Skip buffer register',
                'type': bool,
                'default': False,
            },
            {
                'name': 'verify',
                'msg': 'Verify IO',
                'type': bool,
                'default': False,
            },
            {
                'name': 'batch_size',
                'msg': 'Batch size',
                'type': int,
                'default': 1,
            },
            {
                'name': 'io_type',
                'msg': 'IO type (0-read, 1-write, 2-randread, 3-randwrite)',
                'type': int,
                'default': 1,
            },
            {
                'name': 'duration',
                'msg': 'Duration in seconds',
                'type': int,
                'default': 0,
            },
            {
                'name': 'random_seed',
                'msg': 'Random seed for random read/write',
                'type': int,
                'default': None,
            },
            {
                'name': 'unaligned',
                'msg': 'Use unaligned(4K) random offsets',
                'type': bool,
                'default': False,
            },
            {
                'name': 'random_data',
                'msg': 'Fill IO buffer with random data',
                'type': bool,
                'default': True,
            },
            {
                'name': 'refill_random',
                'msg': 'Refill IO buffer with random data during each write',
                'type': bool,
                'default': False,
            },
            {
                'name': 'alignment',
                'msg': 'Alignment size for random IO',
                'type': int,
                'default': None,
            },
            {
                'name': 'mixed_ratio',
                'msg': 'Mixed read/write percentage in regular batch mode',
                'type': int,
                'default': None,
            },
            {
                'name': 'rdma_url',
                'msg': 'RDMA URL',
                'type': str,
                'default': '',
            },
            {
                'name': 'job_stats',
                'msg': 'Per job statistics',
                'type': bool,
                'default': True,
            }
        ]

    def _configure(self, **kwargs):
        """
        Converts the Jarvis configuration to application-specific configuration.
        E.g., OrangeFS produces an orangefs.xml file.

        :param kwargs: Configuration parameters for this pkg.
        :return: None
        """
        pass

    def start(self):
        """
        Launch an application. E.g., OrangeFS will launch the servers, clients,
        and metadata services on all necessary pkgs.

        :return: None
        """
        use_batch_size = False
        if self.config['xfer_type'] == 3 or self.config['xfer_type'] > 5:
            use_batch_size = True
        cmd = [
            '/usr/local/cuda/gds/tools/gdsio',
            f"-f {self.config['file']}" if self.config['file'] else "",
            f"-D {self.config['directory']}" if self.config['directory'] else "", 
            f"-d {self.config['gpu_index']}",
            f"-n {self.config['numa']}" if self.config['numa'] is not None else "",
            f"-m {self.config['memtype']}",
            f"-w {self.config['threads']}",
            f"-s {self.config['size']}",
            f"-o {self.config['offset']}",
            f"-i {self.config['io_size']}",
            "-p" if self.config['nvlinks'] else "",
            "-b" if self.config['skip_bufregister'] else "",
            "-V" if self.config['verify'] else "",
            f"-x {self.config['xfer_type']}",
            f"-B {self.config['batch_size']}" if use_batch_size else "",
            f"-I {self.config['io_type']}",
            f"-T {self.config['duration']}" if self.config['duration'] > 0 else "",
            f"-k {self.config['random_seed']}" if self.config['random_seed'] is not None else "",
            "-U" if self.config['unaligned'] else "",
            "-R" if self.config['random_data'] else "",
            "-F" if self.config['refill_random'] else "",
            f"-a {self.config['alignment']}" if self.config['alignment'] is not None else "",
            f"-M {self.config['mixed_ratio']}" if self.config['mixed_ratio'] is not None else "",
            f"-P {self.config['rdma_url']}" if self.config['rdma_url'] else "",
            "-J" if self.config['job_stats'] else ""
        ]
        cmd = ' '.join(cmd)
        print(cmd)
        self.gdsio_exec = Exec(cmd,
                MpiExecInfo(nprocs=self.config['nodes'],
                            ppn=1,
                            env=self.env,
                            exec_async=self.config['async'],
                            do_dbg=self.config['do_dbg'],
                            dbg_port=self.config['dbg_port'],
                            collect_output=True))

    def stop(self):
        """
        Stop a running application. E.g., OrangeFS will terminate the servers,
        clients, and metadata services.

        :return: None
        """
        Kill('.*gdsio.*',
             PsshExecInfo(hostfile=self.jarvis.hostfile,
                          env=self.env))

    def clean(self):
        """
        Destroy all data for an application. E.g., OrangeFS will delete all
        metadata and data directories in addition to the orangefs.xml file.

        :return: None
        """
        pass

    def _get_stat(self, stat_dict):
        """
        Get statistics from the application.

        :param stat_dict: A dictionary of statistics.
        :return: None
        """
        stat_dict[f'{self.pkg_id}.gbps'] = self.parse_thrpt()
        stat_dict[f'{self.pkg_id}.start_time'] = self.start_time

    def parse_thrpt(self):
        max_throughput = 0
        if self.gdsio_exec is not None:
            for host, out in self.gdsio_exec.stdout.items():
                match = re.search(r'Throughput:\s*([\d.]+)\s*GiB/sec', out)
                if match:
                    throughput = float(match.group(1))
                    max_throughput = max(max_throughput, throughput)
        return max_throughput
    