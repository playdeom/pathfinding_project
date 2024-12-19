from main import *

if __name__=='__main__':
    
    n=int(input("maze size(only even number and need to bigger than 20): "))
    t=int(input("iterations: "))
    data=dict()
    data['A*']=[]
    data['Flood Fill']=[]
    data['RA(0.5)']=[]
    data['RA(1)']=[]
    data['UFSPS']=[]
    data['MHIP']=[]
    data['PSO']=[]
    data['DFS']=[]
    import tqdm
    for r in tqdm.tqdm(range(0,1000,100)):
        a_star_average_moves = 0
        a_star_average_turns = 0
        a_star_average_time = 0
        flood_average_moves = 0
        flood_average_turns = 0
        flood_average_time = 0
        dfs_average_moves = 0
        dfs_average_turns = 0
        dfs_average_time = 0
        rotation_average_moves_t1 = 0
        rotation_average_turns_t1 = 0
        rotation_average_time_t1 = 0
        rotation_average_moves_t05 = 0
        rotation_average_turns_t05 = 0
        rotation_average_time_t05 = 0
        # rl_average_moves = 0
        # rl_average_turns = 0
        uf_sps_average_moves = 0
        uf_sps_average_turns = 0
        uf_sps_average_time = 0
        mhip_average_moves = 0
        mhip_average_turns = 0
        mhip_average_time = 0
        pso_average_moves = 0
        pso_average_turns = 0
        pso_average_time = 0
        for iter in tqdm.tqdm(range(t)):
            horizontal_walls, vertical_walls = generate_perfect_maze(n)
            introduce_loops(n, horizontal_walls, vertical_walls, loop_factor=r/1000)

            start = (0,0)
            end = (n-1,n-1)

            a_star_path, _ = a_star_search(n, horizontal_walls, vertical_walls, start, end)
            flood_path, _ = flood_fill_search(n, horizontal_walls, vertical_walls, start, end)
            dfs_path_, _ = dfs_search(n, horizontal_walls, vertical_walls, start, end)
            rotation_path_t05, _ = rotation_aware_search(n, horizontal_walls, vertical_walls, start, end, turn_cost=0.5)
            rotation_path_t1, _ = rotation_aware_search(n, horizontal_walls, vertical_walls, start, end, turn_cost=1)
            uf_sps_path, _ = uf_sps_search(n, horizontal_walls, vertical_walls, start, end, turn_cost=0.5)
            mhip_path, _ = mhip_pathfinder_search(n, horizontal_walls, vertical_walls, start, end, turn_cost=0.25)
            pso_path, _ = pso_pathfinding(n, horizontal_walls, vertical_walls, start, end)
            # rl_path, _ = rl_path_search(n, horizontal_walls, vertical_walls, start, end, episodes=2000, alpha=0.1, gamma=0.9, epsilon=0.1)

            a_star_moves, a_star_turns = calculate_moves_and_turns_for_path(a_star_path if a_star_path else [])
            flood_moves, flood_turns = calculate_moves_and_turns_for_path(flood_path if flood_path else [])
            dfs_moves, dfs_turns = calculate_moves_and_turns_for_path(dfs_path_ if dfs_path_ else [])
            rotation_moves_t1, rotation_turns_t1 = calculate_moves_and_turns_for_path(rotation_path_t1 if rotation_path_t1 else [])
            rotation_moves_t05, rotation_turns_t05 = calculate_moves_and_turns_for_path(rotation_path_t05 if rotation_path_t05 else [])
            uf_sps_moves, uf_sps_turns = calculate_moves_and_turns_for_path(uf_sps_path if uf_sps_path else [])
            mhip_moves, mhip_turns = calculate_moves_and_turns_for_path(mhip_path if mhip_path else [])
            pso_moves, pso_turns = calculate_moves_and_turns_for_path(pso_path if pso_path else [])
            # rl_moves, rl_turns = calculate_moves_and_turns_for_path(rl_path if rl_path else [])

            a_star_time = calculate_physical_time_for_path(a_star_path)
            flood_time = calculate_physical_time_for_path(flood_path)
            dfs_time = calculate_physical_time_for_path(dfs_path_)
            rotation_time_t1 = calculate_physical_time_for_path(rotation_path_t1)
            rotation_time_t05 = calculate_physical_time_for_path(rotation_path_t05)
            uf_sps_time = calculate_physical_time_for_path(uf_sps_path)
            mhip_time = calculate_physical_time_for_path(mhip_path)
            pso_time = calculate_physical_time_for_path(pso_path)
            # rl_time = calculate_physical_time_for_path(rl_path)

            a_star_average_moves,a_star_average_turns,a_star_average_time=update_average(a_star_average_moves,a_star_average_turns,a_star_average_time,a_star_moves,a_star_turns,a_star_time)
            flood_average_moves,flood_average_turns,flood_average_time=update_average(flood_average_moves,flood_average_turns,flood_average_time,flood_moves,flood_turns,flood_time)
            dfs_average_moves,dfs_average_turns,dfs_average_time=update_average(dfs_average_moves,dfs_average_turns,dfs_average_time,dfs_moves,dfs_turns,dfs_time)
            rotation_average_moves_t1,rotation_average_turns_t1,rotation_average_time_t1=update_average(rotation_average_moves_t1,rotation_average_turns_t1,rotation_average_time_t1,rotation_moves_t1,rotation_turns_t1,rotation_time_t1)
            rotation_average_moves_t05,rotation_average_turns_t05,rotation_average_time_t05=update_average(rotation_average_moves_t05,rotation_average_turns_t05,rotation_average_time_t05,rotation_moves_t05,rotation_turns_t05,rotation_time_t05)
            uf_sps_average_moves,uf_sps_average_turns,uf_sps_average_time=update_average(uf_sps_average_moves,uf_sps_average_turns,uf_sps_average_time,uf_sps_moves,uf_sps_turns,uf_sps_time)
            mhip_average_moves,mhip_average_turns,mhip_average_time=update_average(mhip_average_moves,mhip_average_turns,mhip_average_time,mhip_moves,mhip_turns,mhip_time)
            pso_average_moves,pso_average_turns,pso_average_time=update_average(pso_average_moves,pso_average_turns,pso_average_time,pso_moves,pso_turns,pso_time)
        
        data["A*"].append([a_star_average_moves/t, a_star_average_turns/t, a_star_average_time/t])
        data["Flood Fill"].append([flood_average_moves/t, flood_average_turns/t, flood_average_time/t])
        data["RA(0.5)"].append([rotation_average_moves_t05/t, rotation_average_turns_t05/t, rotation_average_time_t05/t])
        data["RA(1)"].append([rotation_average_moves_t1/t, rotation_average_turns_t1/t, rotation_average_time_t1/t])
        data["UFSPS"].append([uf_sps_average_moves/t, uf_sps_average_turns/t, uf_sps_average_time/t])
        data["MHIP"].append([mhip_average_moves/t, mhip_average_turns/t, mhip_average_time/t])
        data["PSO"].append([pso_average_moves/t, pso_average_turns/t, pso_average_time/t])
        data["DFS"].append([dfs_average_moves/t, dfs_average_turns/t, dfs_average_time/t])
        
    import pandas as pd

    # Save Moves data
    df_moves = pd.DataFrame({
        "A*": [item[0] for item in data["A*"]],
        "Flood Fill": [item[0] for item in data["Flood Fill"]],
        "RA(0.5)": [item[0] for item in data["RA(0.5)"]],
        "RA(1)": [item[0] for item in data["RA(1)"]],
        "UFSPS": [item[0] for item in data["UFSPS"]],
        "MHIP": [item[0] for item in data["MHIP"]],
        "PSO": [item[0] for item in data["PSO"]],
        "DFS": [item[0] for item in data["DFS"]]
    })
    df_moves.to_csv(f'data/moves.csv', index=False)

    # Save Turns data
    df_turns = pd.DataFrame({
        "A*": [item[1] for item in data["A*"]],
        "Flood Fill": [item[1] for item in data["Flood Fill"]],
        "RA(0.5)": [item[1] for item in data["RA(0.5)"]],
        "RA(1)": [item[1] for item in data["RA(1)"]],
        "UFSPS": [item[1] for item in data["UFSPS"]],
        "MHIP": [item[1] for item in data["MHIP"]],
        "PSO": [item[1] for item in data["PSO"]],
        "DFS": [item[1] for item in data["DFS"]]
    })
    df_turns.to_csv(f'data/turns.csv', index=False)

    # Save Time data
    df_time = pd.DataFrame({
        "A*": [item[2] for item in data["A*"]],
        "Flood Fill": [item[2] for item in data["Flood Fill"]],
        "RA(0.5)": [item[2] for item in data["RA(0.5)"]],
        "RA(1)": [item[2] for item in data["RA(1)"]],
        "UFSPS": [item[2] for item in data["UFSPS"]],
        "MHIP": [item[2] for item in data["MHIP"]],
        "PSO": [item[2] for item in data["PSO"]],
        "DFS": [item[2] for item in data["DFS"]]
    })
    df_time.to_csv(f'data/time.csv', index=False)