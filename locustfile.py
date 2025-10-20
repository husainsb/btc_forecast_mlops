from locust import HttpUser, task, between
import numpy as np
import json

class ModelUser(HttpUser):
    wait_time = between(0.05, 0.1)  # short think time between requests
    headers = {"Content-Type": "application/json"}
    def inputs_data(self):
        #example Input to stress test the API/Inference
        inputs = [[59748.4, 58127.4, 61229. , 57900. ],
                [58118.7, 58076.8, 58890.9, 57686. ],
                [58077.4, 55948. , 58136.7, 55721.6],
                [59748.4, 58127.4, 61229. , 57900. ],
                [58118.7, 58076.8, 58890.9, 57686. ],
                [58077.4, 55948. , 58136.7, 55721.6],
                [59748.4, 58127.4, 61229. , 57900. ],
                [58118.7, 58076.8, 58890.9, 57686. ],
                [58077.4, 55948. , 58136.7, 55721.6],
                [59748.4, 58127.4, 61229. , 57900. ],
                [58118.7, 58076.8, 58890.9, 57686. ],
                [58077.4, 55948. , 58136.7, 55721.6],
                [59748.4, 58127.4, 61229. , 57900. ],
                [58118.7, 58076.8, 58890.9, 57686. ],
                [58077.4, 55948. , 58136.7, 55721.6]]
        
        return inputs
    @task
    def predict(self):
        payload = {"data": self.inputs_data()}
        self.client.post(url="/invocations",data=json.dumps(payload),headers=self.headers)
        #self.client.post(url="",data=json.dumps(payload),headers=self.headers)