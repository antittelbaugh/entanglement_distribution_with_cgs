from qiskit_braket_provider import BraketAwsBackend
import sys

class VerbatimBraketBackend(BraketAwsBackend):
    running=False
    def run(self,*args,**kwargs):
        self.running = True
        job = super().run(*args,verbatim=True,**kwargs)
        self.task_id = job.task_id()
        print(self.task_id)
        sys.stdout.flush()
        return job  
