def dfs_search(n, horizontal_walls, vertical_walls, start, end):
    def can_move(r1, c1, r2, c2):
        if r2 < 0 or r2 >= n or c2 < 0 or c2 >= n:
            return False
        dr, dc = r2 - r1, c2 - c1
        if dr == 1 and dc == 0:
            return not horizontal_walls[r1+1][c1]
        elif dr == -1 and dc == 0:
            return not horizontal_walls[r1][c1]
        elif dr == 0 and dc == 1:
            return not vertical_walls[r1][c1+1]
        elif dr == 0 and dc == -1:
            return not vertical_walls[r1][c1]
        return False

    start_r, start_c = start
    end_r, end_c = end

    visited = [[False]*n for _ in range(n)]
    parent = {}
    visited_states = {(start_r, start_c): (None, None)}

    stack = [(start_r, start_c)]
    visited[start_r][start_c] = True

    while stack:
        r, c = stack.pop()
        if (r, c) == (end_r, end_c):
            path = []
            cur = (r, c)
            while cur != (start_r, start_c):
                path.append(cur)
                cur = parent[cur]
            path.append((start_r, start_c))
            path.reverse()
            return path, visited_states

        for nr, nc in [(r+1,c),(r-1,c),(r,c+1),(r,c-1)]:
            if 0 <= nr < n and 0 <= nc < n and not visited[nr][nc]:
                if can_move(r, c, nr, nc):
                    visited[nr][nc] = True
                    parent[(nr, nc)] = (r, c)
                    if (nr,nc) not in visited_states:
                        visited_states[(nr, nc)] = (r, c)
                    stack.append((nr, nc))

    return None, visited_states
