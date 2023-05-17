import os
from subprocess import Popen, PIPE
from typing import Callable
import time
import numpy as np
from multiprocessing import Process
import threading
import socket
import json

class GPU:
    def __init__(self, ID, uuid, load, memoryTotal, memoryUsed, memoryFree, driver, gpu_name, serial, display_mode, display_active, temp_gpu):
        self.id = ID
        self.uuid = uuid
        self.load = load
        self.memoryUtil = float(memoryUsed)/float(memoryTotal)
        self.memoryTotal = memoryTotal
        self.memoryUsed = memoryUsed
        self.memoryFree = memoryFree
        self.driver = driver
        self.name = gpu_name
        self.serial = serial
        self.display_mode = display_mode
        self.display_active = display_active
        self.temperature = temp_gpu


def safeFloatCast(nbr):
    try:
        number = float(nbr)
    except ValueError:
        number = float('nan')
    return number

class GPUSchedulerServer:
    def __init__(self, socket):
        # Create a UDP socket
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.server_address = ('localhost', socket)
        self.server_socket.bind(self.server_address)
        self.running = True

        self.server_thread = self.start_server()
        self.client_addresses = []

        self.queue = []

    def start_server(self):
        thread = threading.Thread(target=self._run)
        thread.start()
        return thread
    def stop_server(self):
        self.running = False
        self.server_thread.join()
        self.server_socket.close()

    def _run(self):
        while self.running:
            if not self._handle_message():
                print('GPUSCHEDULE ERROR: server could not process message')
        print('GPUSCHEDULE: server stopped')
    
    def _handle_message(self):
        try:
            data, rec_client_address = self.server_socket.recvfrom(1024).decode()
        except:
            return False
        if rec_client_address not in self.client_addresses:
            self.client_addresses.append(rec_client_address)
        received_dict = json.loads(data)

        if received_dict['action'] == 'add_job':
            priority = received_dict['priority']
            for i,j in enumerates(self.queue):
                if j['priority']> priority:
                    continue
            self.queue.append({'c_address': rec_client_address, 'priority': priority})
        


class GPUScheduler():
    _instance = None

    def __new__(self, *args, **kwargs):
        if not self._instance:
            self._instance = super().__new__(self)
        return self._instance
    
    def __init__(self,  visible_ids: list=None,
                        stack_gpu = False,
                            max_usage=0.45,
                            max_memory=0.4,
                        usage_threshold=0.1,
                        memory_threshold=0.1,
                        monitor_duration=4,
                        monitor_frequency=0.5,
                        host_socket=6073,
                        local=True
                ):
        
        self.visible_ids = visible_ids
        self.stack_gpu = stack_gpu
        self.max_usage = max_usage
        self.max_memory = max_memory
        self.usage_threshold = usage_threshold
        self.memory_threshold = memory_threshold
        self.monitor_duration = monitor_duration
        self.monitor_frequency = monitor_frequency

        if not local:
            self.host_socket = host_socket
            self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.client_thread = None
            self.server_address = ('localhost', self.host_socket)
            if not self._is_server_running():
                self.server = GPUSchedulerServer(self.host_socket)
            else:
                self.server = None
        self.local = local
        self.gpu_list = []
        self.gpu_num = 0
        self.updateGPUs()

        self.job_queue = []

        if self.visible_ids is None:
            self.visible_ids = range(len(self.gpu_list))
        else:
            assert all(self.visible_ids < len(self.gpu_list) and self.visible_ids >= 0), "Visible GPU indices are out of range"

    
    
    def register_function(self, function: Callable, insert='last'):
        self._insert_job([function], insert)
        if not self.local:
            self._send_jobs()

    def register_function(self, function: list(Callable), insert='last'):
        self._insert_job(function, insert)

    def execute(self, gpu_id=None):
        visible_gpus = self.visible_ids

        if gpu_id is not None:
            assert gpu_id in self.visible_ids, "Invalid GPU ID"
            visible_gpus = [gpu_id]

        avail_gpus = self.monitor_availability(visible_gpus)
        if len(avail_gpus) == 0:
            return -1

        function = self.job_queue.pop(0)

        
    def monitor_availability(self, ids):
        end_time = time.time() + self.monitor_duration

        load_list = []

        while time.time() < end_time:
            s_1 = time.time()
            gpu_load = self.get_load(ids=ids)
            load_list.append(gpu_load)

            s_2 = time.time()
            time.sleep(self.monitor_frequency-(s_2-s_1))
        load_list = np.stack(load_list)
        usage = np.mean(load_list[...,0], axis=0)
        memUsed = np.mean(load_list[...,1], axis=0)

        usage_avail = np.argwhere(usage <= self.usage_threshold)
        memUsed_avail = np.argwhere(memUsed <= self.memory_threshold)

        available_ids = []
        for i, ind in enumerate(ids):
            if i in usage_avail and i in memUsed_avail:
                available_ids.append(ind)
        return available_ids
           
            


    def updateGPUs(self):
        """
            This functions opens nvidia-smi in a subprocess and retrieve
        """
        output = self._read_nvidia_smi()
        lines = output.split(os.linesep)
        #print(lines)
        self.gpu_num = len(lines)-1
        for g in range(self.gpu_num):
            line = lines[g]
            #print(line)
            vals = line.split(', ')
            #print(vals)
            for i in range(12):
                # print(vals[i])
                if (i == 0):
                    deviceIds = int(vals[i])
                elif (i == 1):
                    uuid = vals[i]
                elif (i == 2):
                    gpuUtil = safeFloatCast(vals[i])/100
                elif (i == 3):
                    memTotal = safeFloatCast(vals[i])
                elif (i == 4):
                    memUsed = safeFloatCast(vals[i])
                elif (i == 5):
                    memFree = safeFloatCast(vals[i])
                elif (i == 6):
                    driver = vals[i]
                elif (i == 7):
                    gpu_name = vals[i]
                elif (i == 8):
                    serial = vals[i]
                elif (i == 9):
                    display_active = vals[i]
                elif (i == 10):
                    display_mode = vals[i]
                elif (i == 11):
                    temp_gpu = safeFloatCast(vals[i]);
            self.gpu_list.append(self._produce_gpu_dict(deviceIds, uuid, gpuUtil, memTotal, memUsed, memFree, driver, gpu_name, serial, display_mode, display_active, temp_gpu))

    def get_load(self, ids):
        output = self._read_nvidia_smi()
        lines = output.split(os.linesep)
        gpu_load = np.zeros((len(ids), 2))

        for id in ids:
            line = lines[id]
            #print(line)
            vals = line.split(', ')

            usage = safeFloatCast(vals[2])/100
            memTotal = safeFloatCast(vals[3])
            memUsed = safeFloatCast(vals[4])
            gpu_load[id][0] = usage
            gpu_load[id][1] =  memUsed / memTotal
        
        return gpu_load

    def _read_nvidia_smi(self):
        try:
            p = Popen(["nvidia-smi", "--query-gpu=index,uuid,utilization.gpu,memory.total,memory.used,memory.free,driver_version,name,gpu_serial,display_active,display_mode,temperature.gpu", "--format=csv,noheader,nounits"], stdout=PIPE)
            stdout, stderror = p.communicate()
        except:
            print("GPU information is unavailable")
            return -1
        output = stdout.decode('UTF-8')
        return output
    
    def _insert_job(self, job: list(Callable), insert):
        if insert == 'last':
            self.job_queue.extend(job)
        elif insert == 'first':
            self.job_queue.insert(0, job)
        elif insert == 'replace':
            self.job_queue = job

    ###--- Client-Server Communication ---###
    # The server can be used to synchronize different instances of GPUScheduler
    
    def _is_server_running(self):
        try:
            socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            socket.settimeout(1)
            socket.bind(('localhost', self.host_socket))
            socket.close()
            # Server is running and the socket was successfully created and bound
            return True
        except socket.error:
            # Server is not running or the socket creation failed
            return False
    def _client_thread(self):
        while True:
            try:
                data, addr = self.client_socket.recvfrom(1024)
                data = json.loads(data.decode('UTF-8'))
                #print(data)
                if 'job_queue' in data:
                    self._insert_job(data['job_queue'], data['insert'])
                if 'gpu_list' in data:
                    self.gpu_list = data['gpu_list']
            except:
                print("Client was not able to receive data")
                continue
    
    def _send_jobs(self, jobs, insert='last'):
        if type(jobs) is not list:
            jobs = [jobs]
        self.client_socket.sendto(json.dumps({'job_queue': jobs, 'insert': insert}).encode(), self.server_address)

    def _send_gpu_list(self):
        self.client_socket.sendto(json.dumps({'gpu_list': self.gpu_list}).encode(), self.server_address)

    def _semd_job(self, job):
        self

    def _produce_gpu_dict(self, ID, uuid, load, memoryTotal, memoryUsed, memoryFree, driver, gpu_name, serial, display_mode, display_active, temp_gpu):
        return {
            'ID': ID,
            'uuid': uuid,
            'load': load,
            'memoryTotal': memoryTotal,
            'memoryUsed': memoryUsed,
            'memoryFree': memoryFree,
            'driver': driver,
            'gpu_name': gpu_name,
            'serial': serial,
            'display_mode': display_mode,
            'display_active': display_active,
            'temp_gpu': temp_gpu
        }