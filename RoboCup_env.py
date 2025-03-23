import gymnasium as gym
from gymnasium import spaces
import numpy as np
import re
import subprocess
import socket
from typing import Optional

UDP_IP = "127.0.0.1"
UDP_PORT_READ_SIM = 5431
UDP_PORT_READ_BASH = 5432
class RoboCupWorldEnv(gym.Env):

    # SITUAZIONE INIZIALE:
    # (cambiato tiro)
    # considero un calcio con 6 fasi, dove la prima è quella per portare la gamba indietro e che non voglio controllare io.
    # 
    # 
    # PLOT: andamento reward, andamento loss actor, andamnto loss critic
    # Guarda scrittura 
    #
    # Per le altre fasi: 
    # assumo che la gamba si possa muovere solo nel piano XZ e ruotare intorno all'asse Y;
    # quindi lo SPAZIO DI STATO dovrebbe essere composto da 3 parametri (x,z,b) per ogni fase (il tempo? per ora costante a 200ms).
    # Lo SPAZIO DI AZIONI è continuo, e rappresenta le velocità. In teoria, il numero di azioni
    # di cui ho bisogno è uguale al numero di parametri nello spazio di stato (+ tempo se non lo considero fissato?).
    # Lo SPAZIO DI OSSERVAZIONI mi viene dalla simulazione, e devo capire cosa mi interessa conoscere.
    #
    # La STEP function credo debba aggiornare la posizione dell'end effector integrando rispetto
    # al tempo lo stato usando le azioni.
    # 
    # La REWARD viene dalla simulazione
    #
    # La RESET function deve resettare lo stato iniziale (quale è per noi lo stato iniziale?)

    # DA AGGIUSTARE
    def __init__(self):
        # Definire lo spazio delle azioni (3 velocità) 
        # Quanto valgono i margini low e high nel mio caso? (max = 4000 mm/s = 4 m/s; min = 0)
        self.kmc_path = '/home/flavio/Scrivania/RoboCup/spqrnao2024/Config/KickEngine/lobKick.kmc'

        self.action_space = spaces.Box(low=-2000, high=2000, shape=(3,), dtype=np.float16)  # REAL
        # self.action_space = spaces.Box(low=-400, high=400, shape=(3,), dtype=np.float16)  # TEST

        self.sock_read_sim = socket.socket(socket.AF_INET,socket.SOCK_DGRAM)  # UDP
        self.sock_read_sim.bind((UDP_IP, UDP_PORT_READ_SIM))
        
        self.sock_read_bash = socket.socket(socket.AF_INET,socket.SOCK_DGRAM)  # UDP
        self.sock_read_bash.bind((UDP_IP, UDP_PORT_READ_BASH))

        # Definire lo spazio degli stati (posizione e rotazioni del piede)
        # Teoricamente lo Stato totale è di 6 elementi (x, y, z, a, b, c)
        # Per ora approssimiamo a 3 (+ tempo? per ora no) per fase, 
        # considerando movimento solo su piano xz e rotazione intorno a y
        # Tempi devono far parte dello stato? Per ora no

        # Forse bastano due fasi per il controllo, perché tanto l'ultima deve 
        # portare al punto di origine
        # self.state = np.zeros(6)  # REAL
        self.state = np.array([-70, -170, -0.1], dtype=np.float16)    # TEST

        # Lo spazio delle osservazioni coincide con lo spazio di stato, poiché
        # il mio stato è completamente osservabile
        # self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float16)   # REAL
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float16)     # TEST
        
        self.done = False


    # Lo step dovrebbe lanciare la simulazione su SimRobot per restituire la reward.
    # Al contempo dovrebbe modificare i valori nel kmc
    def step(self, action):
        # Estrarre le velocità e la durata dall'azione
        # velocities = action[:6]     # REAL
        velocities = action[:3]     # TEST
        
        # print(f"ENV      ->      action:     {action[0]},   {action[1]},    {action[2]}")
        # print(f"ENV      ->      state before:     {self.state[0]},   {self.state[1]},    {self.state[2]}")
        
        duration = 0.2 # secondi, cioè 100 ms
        
        # Aggiornare lo stato del piede in base alle velocità e alla durata
        self.state[0] += velocities[0] * duration
        self.state[1] += velocities[1] * duration
        self.state[2] += velocities[2] /100 * duration
        # print(f"ENV      ->      state after:     {self.state[0]},   {self.state[1]},    {self.state[2]}")
        
        # Modificare i parametri sul kmc
        self.modify_kmc(self.state)

        # Calcolare la ricompensa e verificare se l'episodio è terminato
        # TODO: Bisogna lanciare la simulazione
        result = subprocess.Popen("/home/flavio/Scrivania/RoboCup/spqrnao2024/external_client/run.sh")

        reward = self.calculate_reward()
        self.done = self.check_done()
        
        data, addr = self.sock_read_bash.recvfrom(1024)  # buffer size is 1024 bytes
        print("ENV  ->   received message: %s" % data.decode('utf-8'))
        
        # Restituire lo stato, la ricompensa, done e info (vuoto in questo caso)
        return self.state, reward, self.done, False, {}

    # Resettare lo stato iniziale
    # lo stato lo resetto a dove si trova l'e-e dopo la seconda fase 
    # (perché io poi vado a modificare la terza e quarta fase,
    # cioè l'ultima di slancio indietro e quell del calcio)

    # INFO: guardare l'esempio dell'environment del pendolo per vedere quando non ci sono 
    # compatibilità di argomenti o cose simili
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
    
        # REAL
        # manca il reset dello stato
        # obs = np.array([-15, -230, 0, -70, -170, -0.1], dtype=np.float16) # x1, z1, b1, x2, z2, b2
        
        # TEST
        self.state = np.array([-70, -170, -0.1], dtype=np.float16)
        obs = self.state       # x1, z1, b1, x2, z2, b2
        self.done = False
        return obs, {}

    # NON necessario
    def render(self, mode='human'):
        pass

    # Ricompensa: viene passata direttamente dalla simulazione al server
    def calculate_reward(self):
        data, addr = self.sock_read_sim.recvfrom(1024)  # buffer size is 1024 bytes
        # print("ENV  ->   received reward: %s" % data.decode('utf-8'))
        reward = float(data.decode('utf-8'))
        print("ENV  ->   received reward:  ", reward)
        return reward

    # Quando l'episodio termina
    def check_done(self):
        done = True
        return done

    # Chiude tutte le connessioni UDP aperte   
    def close(self):
        self.sock_read_sim.close()
        self.sock_read_bash.close()
        result = subprocess.run("/home/flavio/Scrivania/RoboCup/spqrnao2024/external_client/kill_pid.sh")

    # Sembra funzionare
    def modify_kmc(self, state):
        line_to_modify_1 = 82                      # linea di rightFootTra1 della terza fase
        line_to_modify_2 = 106                     # linea di rightFootTra1 della quarta fase
        with open(self.kmc_path, 'r+') as file:
            lines = file.readlines()
        
        # x1, z1, b1, x2, z2, b2 = state        # REAL
        x2, z2, b2 = state        # REAL
        
        # line1_tra1 = lines[line_to_modify_1]
        # line1_tra1 = re.sub(r'x\s*=\s*-?\d+(\.\d+)?', f'x = {x1}', line1_tra1)
        # line1_tra1 = re.sub(r'z\s*=\s*-?\d+(\.\d+)?', f'z = {z1}', line1_tra1)

        # line1_tra2 = lines[line_to_modify_1+1]
        # line1_tra2 = re.sub(r'x\s*=\s*-?\d+(\.\d+)?', f'x = {x1}', line1_tra2)
        # line1_tra2 = re.sub(r'z\s*=\s*-?\d+(\.\d+)?', f'z = {z1}', line1_tra2)
    
        # line1_rot1 = lines[line_to_modify_1 + 2]
        # line1_rot1 = re.sub(r'y\s*=\s*-?\d+(\.\d+)?', f'y = {b1}', line1_rot1)
        
        # line1_rot2 = lines[line_to_modify_1 + 3]
        # line1_rot2 = re.sub(r'y\s*=\s*-?\d+(\.\d+)?', f'y = {b1}', line1_rot2)

        line2_tra1 = lines[line_to_modify_2]
        line2_tra1 = re.sub(r'x\s*=\s*-?\d+(\.\d+)?', f'x = {x2}', line2_tra1)
        line2_tra1 = re.sub(r'z\s*=\s*-?\d+(\.\d+)?', f'z = {z2}', line2_tra1)
    
        line2_tra2 = lines[line_to_modify_2 + 1]
        line2_tra2 = re.sub(r'x\s*=\s*-?\d+(\.\d+)?', f'x = {x2}', line2_tra2)
        line2_tra2 = re.sub(r'z\s*=\s*-?\d+(\.\d+)?', f'z = {z2}', line2_tra2)
    
        line2_rot1 = lines[line_to_modify_2 + 2]
        line2_rot1 = re.sub(r'y\s*=\s*-?\d+(\.\d+)?', f'y = {b2}', line2_rot1)

        line2_rot2 = lines[line_to_modify_2 + 3]
        line2_rot2 = re.sub(r'y\s*=\s*-?\d+(\.\d+)?', f'y = {b2}', line2_rot2)

        # Le due righe di ogni soecifica dell'e-e le considero uguali
        # lines[line_to_modify_1] = line1_tra1
        # lines[line_to_modify_1 + 1] = line1_tra2
        # lines[line_to_modify_1 + 2] = line1_rot1
        # lines[line_to_modify_1 + 3] = line1_rot2
        
        lines[line_to_modify_2] = line2_tra1
        lines[line_to_modify_2 + 1] = line2_tra2
        lines[line_to_modify_2 + 2] = line2_rot1
        lines[line_to_modify_2 + 3] = line2_rot2

        with open(self.kmc_path, 'w') as file:
            file.writelines(lines)

