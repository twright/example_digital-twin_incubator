import time
# import sys
try:
    from communication.shared.connection_parameters import *
    from communication.shared.protocol import *
    from communication.server.rabbitmq import Rabbitmq
except:
    raise


class ControllerPhysical():
    def __init__(self, desired_temperature=35.0, lower_bound=5, heating_time=3, heating_gap=2):
        self.temperature_desired = desired_temperature
        self.lower_bound = lower_bound
        self.heating_time = heating_time
        self.heating_gap = heating_gap

        self.box_air_temperature = None
        self.room_temperature = None
        self.sensor1_reading = None
        self.sensor2_reading = None
        self.sensor3_reading = None

        self.heater_ctrl = None
        self.current_state = "CoolingDown"
        self.next_time = -1.0

        self.rabbitmq = Rabbitmq()
        self.state_queue_name = 'state'
        self.message = None

        print("Before running actually, please make sure that the low_level_deriver_server is running")
        print("And using command (sudo rabbitmqctl list_queues) to check the (heater_control) queue and (fan_control) queue exist.")

    def _message_decode(self):
        if self.message is not None:
            self.sensor1_reading = float(self.message['t1'])
            self.sensor2_reading = float(self.message['t2'])
            self.sensor3_reading = float(self.message['t3'])
            self.box_air_temperature = (self.sensor2_reading + self.sensor3_reading)/2
            self.room_temperature = self.sensor1_reading

    def safe_protocol(self):
        self.rabbitmq.send_message(routing_key=ROUTING_KEY_FAN, message='False')
        self.rabbitmq.send_message(routing_key=ROUTING_KEY_HEATER, message='False')

    # this can be further used to self-adaption
    def change_controller_parameters(self, desired_temperature=35.0, lower_bound=5, heating_time=0.2, heating_gap=0.3):
        self.temperature_desired = desired_temperature
        self.lower_bound = lower_bound
        self.heating_time = heating_time
        self.heating_gap = heating_gap
        print(f"Controller parameters have been changed to"
              f"\ntemperature_desired:{self.temperature_desired}"
              f"\nlower_bound:{self.lower_bound}"
              f"\nheating_time:{self.heating_time}"
              f"\nheating_gap:{self.heating_gap}"
              )

    def setup(self):
        self.rabbitmq.connect_to_server()
        self.safe_protocol()
        self.rabbitmq.send_message(routing_key=ROUTING_KEY_FAN, message='True')

    def get_state_message(self):
        self.message = self.rabbitmq.get_message(queue_name=self.state_queue_name, binding_key=ROUTING_KEY_STATE)
        if self.message is None:
            #print("No state information")
            return None
        else:
            self._message_decode()
            return 0

    def ctrl_step(self):
        try:
            if self.current_state == "CoolingDown":
                # print("current state is: CoolingDown")
                self.heater_ctrl = False
                if self.box_air_temperature <= self.temperature_desired - self.lower_bound:
                    self.current_state = "Heating"
                    self.heater_ctrl = True
                    self.next_time = time.time() + self.heating_time
                    return
            if self.current_state == "Heating":
                # print("current state is: Heating")
                self.heater_ctrl = True
                if 0 < self.next_time <= time.time():
                    self.current_state = "Waiting"
                    self.heater_ctrl = False
                    self.next_time = time.time() + self.heating_gap
                return
            if self.current_state == "Waiting":
                # print("current state is: Waiting")
                self.heater_ctrl = False
                if 0 < self.next_time <= time.time():
                    if self.box_air_temperature <= self.temperature_desired:
                        self.current_state = "Heating"
                        self.heater_ctrl = True
                        self.next_time = time.time() + self.heating_time
                    else:
                        self.current_state = "CoolingDown"
                        self.heater_ctrl = False
                        self.next_time = -1
                return
        except:
            print("Check the Values of all parameters. Did you get message first?")
            raise

    def cleanup(self):
        self.safe_protocol()
        self.rabbitmq.close()

    def start_control(self):
        try:
            self.setup()
            while True:
                print("Start getting message")
                while True:
                    if self.get_state_message() is not None:
                        break
                    pass
                print(f"Time: {time.ctime(self.message['time'])}\n"
                      f"Box air temperature is: {self.box_air_temperature}\n"
                      f"Heat state: {self.message['heater_on']}\n"
                      f"Fan state: {self.message['fan_on']}\n"
                      f"current state: {self.current_state}\n")
                if self.box_air_temperature >= 50:
                    print("Cleaning up")
                    self.cleanup()
                    break
                self.ctrl_step()
                self.rabbitmq.send_message(routing_key=ROUTING_KEY_HEATER,
                                           message=self.heater_ctrl
                                           )
        except:
            self.cleanup()
            raise


if __name__ == '__main__':
    idx = int(input("0-For testing the functionality\n1-For control: "))
    if idx == 1:
        desired_temperature = float(input("Please input desired temperature: "))
        controller = ControllerPhysical(desired_temperature=desired_temperature)
        #controller.rabbitmq.connect_to_server()
        #controller.rabbitmq.declare_queue(queue_name=controller.state_queue_name, routing_key=ROUTING_KEY_STATE)
        #time.sleep(5)
        #controller.get_state_message()
        #time.sleep(5)
        print(f"sensor3 readings:{controller.sensor3_reading}\n"
              f"sensor2 readings:{controller.sensor2_reading}\n"
              f"sensor1 readings:{controller.sensor1_reading}\n"
              f"box air temperature is:{controller.box_air_temperature}\n"
              f"room temperature is:{controller.room_temperature}\n")
        #if controller.sensor3_reading is None:
            #print("None state message get")
        #else:
        controller.start_control()
    else:
        ctrl = ControllerPhysical()
        # ctrl.start_control()
        # ctrl.rabbitmq.declare_queue(queue_name=ctrl.state_queue_name, routing_key=ROUTING_KEY_STATE)
        ctrl.setup()
        ctrl.get_state_message()
        ctrl.rabbitmq.send_message(routing_key=ROUTING_KEY_HEATER, message=True)
        time.sleep(20)
        ctrl.rabbitmq.send_message(routing_key=ROUTING_KEY_HEATER, message=False)
        ctrl.get_state_message()
        print(ctrl.message)
        print(f"sensor3 readings:{ctrl.sensor3_reading}\n"
              f"sensor2 readings:{ctrl.sensor2_reading}\n"
              f"sensor1 readings:{ctrl.sensor1_reading}\n"
              f"box air temperature is:{ctrl.box_air_temperature}\n"
              f"room temperature is:{ctrl.room_temperature}\n")
        ctrl.cleanup()