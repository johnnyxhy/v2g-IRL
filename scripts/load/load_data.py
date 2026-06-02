import irl.dataset.expert_loader_airl_discrete as airl_discrete
import irl.dataset.expert_loader_airl_continuous as airl_continuous
import irl.dataset.expert_loader_continuous as linear_continuous
import irl.dataset.expert_loader_discrete as linear_discrete
import irl.dataset.expert_loader_continuous_no_profit as linear_continuous_no_profit
import irl.dataset.expert_loader_simple as linear_simple 
import irl.dataset.expert_loader_deep_discrete as deep_discrete
import irl.dataset.expert_loader_deep_continuous as deep_continuous

def load_expert_data(args):
    loader = args['expert_loader']
    input_file = args['input_file']
    output_file = args.get('output_file')
    if loader == 'airl_discrete':
        return airl_discrete.load_trajectories(input_file, output_file)
    elif loader == 'airl_continuous':
        return airl_continuous.load_trajectories(input_file, output_file)
    elif loader == 'linear_continuous':
        return linear_continuous.load_trajectories(input_file, output_file)
    elif loader == 'linear_discrete':
        return linear_discrete.load_trajectories(input_file, output_file)
    elif loader == 'linear_continuous_no_profit':
        return linear_continuous_no_profit.load_trajectories(input_file, output_file)
    elif loader == 'linear_simple':
        return linear_simple.load_trajectories(input_file, output_file)
    elif loader == 'deep_discrete':
        return deep_discrete.load_trajectories(input_file, output_file)
    elif loader == 'deep_continuous':
        return deep_continuous.load_trajectories(input_file, output_file)
    
# State arguments

args = {
    'expert_loader': 'deep_discrete',
    'input_file': 'data/EVDataset_discrete_special_highbat.csv',
    'output_file': 'data/processed_trajectories_deep_discrete_special_lowbat.json',
}

if __name__ == "__main__":
    episodes = load_expert_data(args)