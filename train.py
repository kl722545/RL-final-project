import Environment
import Codec
import Agent

codec = Codec.Codec()
RL_environment = Environment.HomeWorld()
agent = Agent.Agent()
RL_environment.new_game()
while True:
    state = RL_environment.get_current_state()
    if RL_environment.if_finished():
        break
    input_command = input('What do you want to do? : ')
    actions = input_command.strip().split(" ")
    reward = RL_environment.do_action(*tuple(actions))
    print("reward : {0}\n".format(reward))


    
