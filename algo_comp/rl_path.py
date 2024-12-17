import random

def rl_path_search(n, horizontal_walls, vertical_walls, start, end, episodes=2000, alpha=0.1, gamma=0.9, epsilon=0.1):
    start_r, start_c = start
    end_r, end_c = end

    # 가능한 행동: 상(0), 우(1), 하(2), 좌(3)
    actions = [( -1, 0), (0, 1), (1, 0), (0, -1)]

    # 상태의 범위: (0 <= r,c < n)
    # Q 테이블: dict key: (r,c,action), value: float
    Q = {}

    def can_move(r1, c1, r2, c2):
        if r2 < 0 or r2 >= n or c2 < 0 or c2 >= n:
            return False
        dr, dc = r2 - r1, c2 - c1
        if dr == 1 and dc == 0:   # 아래
            return not horizontal_walls[r1+1][c1]
        elif dr == -1 and dc == 0: # 위
            return not horizontal_walls[r1][c1]
        elif dr == 0 and dc == 1:  # 오른쪽
            return not vertical_walls[r1][c1+1]
        elif dr == 0 and dc == -1: # 왼쪽
            return not vertical_walls[r1][c1]
        return False

    def get_actions(r, c):
        # 가능한 행동 반환
        valid_actions = []
        for i, (dr,dc) in enumerate(actions):
            nr, nc = r+dr, c+dc
            if can_move(r,c,nr,nc):
                valid_actions.append(i)
        return valid_actions

    def get_Q(r,c,a):
        return Q.get((r,c,a),0.0)

    def set_Q(r,c,a,value):
        Q[(r,c,a)] = value

    # 보상 함수
    def get_reward(r, c):
        if (r,c) == (end_r,end_c):
            return 100.0
        return -1.0

    # 학습
    for _ in range(episodes):
        r, c = start_r, start_c
        visited_set = set() # 상태 방문 기록
        step_count = 0
        while True:
            step_count += 1
            # 종료 조건: 너무 오래 걸리면 중단
            if step_count > n*n*2:
                break

            if (r,c) == (end_r,end_c):
                # 목표 도달
                break

            valid_actions = get_actions(r,c)
            if len(valid_actions) == 0:
                # 막힌 경우 penalty
                # 막힘상황에서는 랜덤으로 못가도 탈출. 여기서는 -10 보상 가정
                # 하지만 여기서는 그냥 중단 처리(episode 종료)
                break

            # eps-greedy
            if random.random() < epsilon:
                a = random.choice(valid_actions)
            else:
                # 최대 Q값 액션 선택
                q_values = [(get_Q(r,c,act), act) for act in valid_actions]
                q_values.sort(key=lambda x:x[0], reverse=True)
                a = q_values[0][1]

            # 상태 전이
            dr,dc = actions[a]
            nr, nc = r+dr, c+dc
            reward = get_reward(nr,nc)

            # Q 업데이트
            q_current = get_Q(r,c,a)
            next_valid_actions = get_actions(nr,nc)
            if len(next_valid_actions) > 0:
                q_next_max = max([get_Q(nr,nc,act) for act in next_valid_actions])
            else:
                q_next_max = 0.0

            new_q = q_current + alpha*(reward + gamma*q_next_max - q_current)
            set_Q(r,c,a,new_q)

            r,c = nr,nc

    # 학습 후, start에서 end까지 Q 기반 탐색
    # 최적 경로 복원: start에서 시작하여 Q값 최대인 방향 선택
    # 경로가 없거나 루프에 빠질 경우 대비 최대 스텝 제한
    path = []
    visited_states = {}
    r, c = start_r, start_c
    visited_states[(r,c)] = (None,None)
    path.append((r,c))
    step_limit = n*n*4
    steps = 0

    while (r,c) != (end_r,end_c) and steps<step_limit:
        steps+=1
        valid_actions = get_actions(r,c)
        if len(valid_actions) == 0:
            # 경로 단절
            path = None
            break
        # Q값 최대 액션
        q_values = [(get_Q(r,c,act), act) for act in valid_actions]
        q_values.sort(key=lambda x:x[0], reverse=True)
        a = q_values[0][1]
        dr,dc = actions[a]
        nr,nc = r+dr,c+dc
        path.append((nr,nc))
        visited_states[(nr,nc)] = (r,c)
        r,c = nr,nc

    if (r,c) != (end_r,end_c):
        path = None

    return path, visited_states
