
import json
from datasets import load_dataset

from tqdm import tqdm

import random

SYSTEM_PROMPT = """You are tasked with controlling a drone swarm based on user commands.
    Currently, we have 3 drones. They fly in a cage of size 1.5x1.5x1.5m.
    The current position of the drones is {'red': [1.0, 1.0, 1.0], 'green': [1.0, 0.0, 1.0], 'blue': [1.0, -1.0, 1.0]}.
    You can control the drones by returning exactly such a python dictionary, giving the next position of the drones. x-direction is forward, y-direction is left, and z-direction is up. Try to keep a distance of 0.5m between the drones.
    Return me only the dictionary, no comments, not explanations.
    Assume every answer you give me is executed, i.e., the current state the user sees is your last answer."""
    

def generate_directions(current_position: dict) -> list:
    distance = random.uniform(0.5, 1.0)
    distance = round(distance, 1)
    num_drones = random.randint(1, 3)
    direction = random.sample(["up", "down", "forward", "backward", "left", "right"], num_drones)
    drones = random.sample(["red", "green", "blue", "all", ""], num_drones)

    all_in_drones = False
    c = None
    if "all" in drones or "" in drones:
        all_in_drones = True
        c = "all" if "all" in drones else ""
        drones = ["red", "green", "blue"]

    new_position = current_position.copy()
    for drone in drones:
        if direction[0] == "up":
            new_position[drone][2] += distance
        elif direction[0] == "down":
            new_position[drone][2] -= distance
        elif direction[0] == "forward":
            new_position[drone][0] += distance
        elif direction[0] == "backward":
            new_position[drone][0] -= distance
        elif direction[0] == "left":
            new_position[drone][1] += distance
        elif direction[0] == "right":
            new_position[drone][1] -= distance
    
    command = ""
    for drone, dir in zip(drones, direction):
        command += f"{drone} {dir}, "
    if all_in_drones:
        command = f"{c} {direction[0]}, "
    command = command[:-2]  # remove last comma and space
    for k in new_position.keys():
        new_position[k] = [round(x, 1) for x in new_position[k]]
    return command, new_position


def generate_triangle(current_position: dict) -> list:
    base = {"red": [1.0, 0.0, 1.0], "blue": [0.0, -1.0, 1.0], "green": [0.0, 1.0, 1.0]}

    size = random.choice(["small", "large", "big", ""])

    scaling_borders = {"small": (0.5, 0.8), "large": (1.2, 1.5), "big": (1.2, 1.5), "": (0.8, 1.2)}
    scaling_factor = random.uniform(*scaling_borders[size]) 

    new_position = {}
    for drone in base.keys():
        new_position[drone] = [round(base[drone][0]*scaling_factor, 1), round(base[drone][1]*scaling_factor, 1), 1.0]

    command = random.choice([f"form a {size} triangle", f"a {size} triangle", f"in a {size} triangle", f"make a {size} triangle", f"fly in a {size} triangle"])

    return command, new_position

def generate_line(current_position: dict) -> tuple:
    base = {"red": [0.5, 0.0, 1.0], "green": [1.0, 0.0, 1.0], "blue": [1.5, 0.0, 1.0]}
    
    directions = [
        "forward to backward",
        "backward to forward",
        "left to right",
        "right to left",
        "forward left to backward right",
        "backward right to forward left",
        "upper left to lower right",
        "lower right to upper left",
        "lower left to upper right",
        "upper right to lower left"
    ]
    
    direction = random.choice(directions)

    length = random.choice(["short", "long", ""])
    length_borders = {"short": (0.5, 0.8), "long": (1.2, 1.5), "": (0.8, 1.2)}
    length_factor = random.uniform(*length_borders[length])
    
    new_position = {}
    if direction == "forward to backward":
        new_position = {"red": [-1.0, 0.0, 1.0], "green": [0.0, 0.0, 1.0], "blue": [1.0, 0.0, 1.0]}
    elif direction == "backward to forward":
        new_position = {"red": [1.0, 0.0, 1.0], "green": [0.0, 0.0, 1.0], "blue": [-1.0, 0.0, 1.0]}
    elif direction == "left to right":
        new_position = {"red": [0.0, -1.0, 1.0], "green": [0.0, 0.0, 1.0], "blue": [0.0, 1.0, 1.0]}
    elif direction == "right to left":
        new_position = {"red": [0.0, 1.0, 1.0], "green": [0.0, 0.0, 1.0], "blue": [0.0, -1.0, 1.0]}
    elif direction == "forward left to backward right":
        new_position = {"red": [-1.0, -1.0, 1.0], "green": [0.0, 0.0, 1.0], "blue": [1.0, 1.0, 1.0]}
    elif direction == "backward right to forward left":
        new_position = {"red": [1.0, 1.0, 1.0], "green": [0.0, 0.0, 1.0], "blue": [1.0, -1.0, 1.0]}
    elif direction == "upper left to lower right":
        new_position = {"red": [-1.0, 1.0, 1.7], "green": [0.0, 0.0, 1.0], "blue": [1.0, -1.0, 0.3]}
    elif direction == "lower right to upper left":
        new_position = {"red": [1.0, -1.0, 0.3], "green": [0.0, 0.0, 1.0], "blue": [-1.0, 1.0, 1.7]}
    elif direction == "lower left to upper right":
        new_position = {"red": [-1.0, -1.0, 0.3], "green": [0.0, 0.0, 1.0], "blue": [1.0, 1.0, 1.7]}
    elif direction == "upper right to lower left":
        new_position = {"red": [1.0, 1.0, 1.7], "green": [0.0, 0.0, 1.0], "blue": [-1.0, -1.0, 0.3]}
    else:
        assert False, f"Invalid direction {direction}"

    for k in new_position.keys():
        new_position[k] = [round(x*length_factor, 1) for x in new_position[k]]
        new_position[k][2] = 1.0  # keep z the same
    
    command = random.choice([f"form a {length} line {direction}", f"in a {length} line {direction}", f"make a {length} line {direction}", f"fly in a {length} line {direction}", f"a {length} line {direction}"])
    
    return command, new_position


def swap_positions(current_position: dict) -> tuple:
    drones = list(current_position.keys())
    drone1, drone2 = random.sample(drones, 2)
    
    new_position = current_position.copy()
    new_position[drone1], new_position[drone2] = new_position[drone2], new_position[drone1]
    
    command = f"swap {drone1} and {drone2}"
    
    return command, new_position


def generate_corner_movement(current_position: dict) -> tuple:
    corners = {
        "left front corner": [-1.0, 1.0, 1.0],
        "left lower corner": [-1.0, 1.0, 0.3],
        "right front corner": [1.0, 1.0, 1.0],
        "right lower corner": [1.0, 1.0, 0.3],
        "middle": [0.0, 1.0, 1.0]
    }
    
    num_drones = random.randint(1, 3)
    drones = random.sample(["red", "green", "blue"], num_drones)
    
    new_position = {}
    corner_assignments = {}
    
    for drone in drones:
        corner = random.choice(list(corners.keys()))
        new_position[drone] = corners[corner]
        corner_assignments[drone] = corner
    
    command = ", ".join([f"{drone} to the {corner_assignments[drone]}" for drone in drones])
    command = f"move {command}"
    
    return command, new_position


GENERATORS = [generate_directions, generate_triangle, generate_line, swap_positions]


def generate_dataset(name, num_samples=1000):
    dataset = []

    for _ in tqdm(range(num_samples)):
        messages = {"messages": [{"role": "system", "content": SYSTEM_PROMPT}]}
        current_position = {'red': [1.0, 1.0, 1.0], 'green': [1.0, 0.0, 1.0], 'blue': [1.0, -1.0, 1.0]}
        for _ in range(7):
            generator = random.choice(GENERATORS)
            command, new_position = generator(current_position)
            messages["messages"].append({"role": "user", "content": command})
            messages["messages"].append({"role": "assistant", "content": str(new_position)})
            current_position = new_position
            print(current_position)     
        dataset.append(messages)

    dataset = {"data": dataset}
    with open(f'/data/datasets/swarm/{name}.json', 'w') as f:
        json.dump(dataset, f, indent=2)




if __name__ == "__main__":
    generate_dataset("train", num_samples=1000)
    generate_dataset("eval", num_samples=200)
    dataset = load_dataset("json", data_files="/data/datasets/swarm/train.json", field="data") 
    print(dataset["train"][0])

    # llm = LLM(model_name="qwen", use_cluster=False)
    # print(llm.chat(dataset["train"][0], max_new_tokens=100))
