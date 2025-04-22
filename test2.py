import pygame
import numpy as np
import math
import random
import time
from collections import deque

# --- Constants ---
WIDTH, HEIGHT = 800, 700
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (200, 0, 0)
ORANGE = (255, 165, 0)
BLUE = (0, 0, 200)
GREEN = (0, 150, 0)
YELLOW = (200, 200, 0)
GRAY = (128, 128, 128)
LIGHT_GRAY = (200, 200, 200)

# Face mapping: (axis index, direction)
# Used for identifying faces and layers for moves
FACE_U = (1, 1)  # Positive Y (Top)
FACE_D = (1, -1) # Negative Y (Bottom)
FACE_L = (0, -1) # Negative X (Left)
FACE_R = (0, 1)  # Positive X (Right)
FACE_F = (2, 1)  # Positive Z (Front)
FACE_B = (2, -1) # Negative Z (Back)

FACES = {
    'U': FACE_U, 'D': FACE_D, 'L': FACE_L,
    'R': FACE_R, 'F': FACE_F, 'B': FACE_B
}

# Colors associated with the *center* cubie of each face (initial state)
FACE_COLORS = {
    'U': WHITE, 'D': YELLOW, 'L': ORANGE,
    'R': RED, 'F': GREEN, 'B': BLUE
}

# Axis vectors for rotations
AXIS_X = np.array([1., 0., 0.])
AXIS_Y = np.array([0., 1., 0.])
AXIS_Z = np.array([0., 0., 1.])

# Map face names to rotation axes and angles for moves
# Format: (axis_vector, angle_multiplier)
# Angle multiplier is combined with direction and clockwise status
MOVE_DEFINITIONS = {
    'U': (AXIS_Y, 1), 'D': (AXIS_Y, -1),
    'L': (AXIS_X, -1), 'R': (AXIS_X, 1),
    'F': (AXIS_Z, 1), 'B': (AXIS_Z, -1),
}

# --- Helper Functions ---

def create_rotation_matrix(axis, angle):
    """ Creates a 3D rotation matrix using axis-angle representation """
    axis = np.asarray(axis)
    # Normalize axis (important!)
    norm = np.linalg.norm(axis)
    if norm == 0: return np.identity(3) # No rotation for zero axis
    axis = axis / norm

    a = math.cos(angle / 2.0)
    b, c, d = -axis * math.sin(angle / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                     [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                     [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])

def project(point_3d, fov=256, viewer_distance=4):
    """ Projects a 3D point to 2D using perspective projection """
    x, y, z = point_3d
    # Basic check to avoid division by zero or projection inversion
    if viewer_distance + z <= 1e-6:
        return np.array([-1000, -1000]) # Point is behind or too close

    factor = fov / (viewer_distance + z)
    x_2d = x * factor + WIDTH / 2
    y_2d = -y * factor + HEIGHT / 2 # Negative y for Pygame screen coords
    return np.array([x_2d, y_2d])

# --- Cubie Class ---
class Cubie:
    """ Represents a single small cube (cubie) within the Rubik's Cube """
    def __init__(self, x, y, z, size=0.95):
        # Center position of the cubie in logical coordinates (-1, 0, 1)
        self.logical_pos = np.array([x, y, z])
        # Current world position (starts same as logical, changes with rotation)
        self.pos = np.array([float(x), float(y), float(z)])
        # Half-size for vertex calculation
        self.half_size = size / 2.0
        # Define the 8 vertices relative to the center position (local space)
        self.vertices_local = np.array([
            [-1, -1, -1], [ 1, -1, -1], [ 1,  1, -1], [-1,  1, -1], # Back face vertices (z=-1)
            [-1, -1,  1], [ 1, -1,  1], [ 1,  1,  1], [-1,  1,  1]  # Front face vertices (z=1)
        ]) * self.half_size
        # Store the current vertices in world space (relative to world origin)
        self.vertices_world = self.vertices_local + self.pos

        # Define the faces using vertex indices (Corrected Winding Order)
        # Indices reference self.vertices_local/self.vertices_world
        # Order should be counter-clockwise when viewed from *outside* the cube
        self.faces_indices = [
            (3, 2, 1, 0), # Back (-Z face) Looking from +Z
            (4, 5, 6, 7), # Front (+Z face) Looking from -Z
            (0, 4, 7, 3), # Left (-X face) Looking from +X
            (1, 2, 6, 5), # Right (+X face) Looking from -X (Corrected)
            (7, 6, 2, 3), # Top (+Y face) Looking from -Y (Corrected)
            (0, 1, 5, 4)  # Bottom (-Y face) Looking from +Y
        ]
        # Store initial orientation vectors (normals pointing outwards)
        # These vectors define which color belongs to which physical face initially
        self.initial_normals = np.array([
            [ 0,  0, -1], # Back
            [ 0,  0,  1], # Front
            [-1,  0,  0], # Left
            [ 1,  0,  0], # Right
            [ 0,  1,  0], # Top
            [ 0, -1,  0]  # Bottom
        ])
        # Store the current orientation of these faces in world space
        self.current_normals = self.initial_normals.copy()
        # Assign face colors based on initial orientation
        self.face_colors = self.assign_colors()

    def assign_colors(self):
        """ Assigns colors to the physical faces based on their initial normal vectors """
        colors = [BLACK] * 6
        color_map = {
            (0, 0, -1): FACE_COLORS['B'], (0, 0, 1): FACE_COLORS['F'],
            (-1, 0, 0): FACE_COLORS['L'], (1, 0, 0): FACE_COLORS['R'],
            (0, 1, 0): FACE_COLORS['U'], (0, -1, 0): FACE_COLORS['D'],
        }
        for i, normal in enumerate(self.initial_normals):
            # Use tuple of rounded ints for dictionary key lookup
            normal_tuple = tuple(int(round(c)) for c in normal)
            if normal_tuple in color_map:
                colors[i] = color_map[normal_tuple]
        return colors

    def get_visible_face_colors(self):
        """ Returns a dictionary mapping world-space normal vectors to colors """
        visible_colors = {}
        for i, color in enumerate(self.face_colors):
            if color != BLACK:
                 # Use tuple of rounded ints for dictionary key
                normal_tuple = tuple(int(round(c)) for c in self.current_normals[i])
                visible_colors[normal_tuple] = color
        return visible_colors

    def rotate(self, rotation_matrix, center=np.array([0.,0.,0.])):
        """ Rotates the cubie's position, vertices, and normals around a center point """
        # Rotate position
        self.pos = np.dot(self.pos - center, rotation_matrix.T) + center
        # Rotate world vertices
        self.vertices_world = np.dot(self.vertices_world - center, rotation_matrix.T) + center
        # Rotate normal vectors
        self.current_normals = np.dot(self.current_normals, rotation_matrix.T)

    def draw(self, screen, rotation_matrix_cam):
        """ Draws the cubie on the screen """
        # Rotate vertices and normals based on camera angle for projection and culling
        rotated_vertices_cam = np.dot(self.vertices_world, rotation_matrix_cam.T)
        rotated_normals_cam = np.dot(self.current_normals, rotation_matrix_cam.T)

        # Project vertices to 2D
        projected_vertices = [project(v) for v in rotated_vertices_cam]

        # Calculate face data for drawing
        face_data = []
        for i, indices in enumerate(self.faces_indices):
            if self.face_colors[i] != BLACK: # Only process colored faces
                # Get the normal vector in camera space for this face
                normal_cam = rotated_normals_cam[i]

                # --- Back-face Culling ---
                # Check if the face normal points towards the camera (negative Z in camera space)
                # A small negative threshold helps catch faces near edge-on
                if normal_cam[2] < -0.01:
                    face_vertices_cam = [rotated_vertices_cam[j] for j in indices]
                    # Calculate average Z depth in camera space for Painter's Algorithm
                    avg_z = sum(v[2] for v in face_vertices_cam) / 4.0

                    face_data.append({
                        'indices': indices,
                        'color': self.face_colors[i],
                        'avg_z': avg_z,
                        'projected': [projected_vertices[j] for j in indices]
                    })

        # Sort faces by average Z-depth (Painter's Algorithm - draw farther first)
        face_data.sort(key=lambda f: f['avg_z'], reverse=True)

        # Draw sorted, visible faces
        for face in face_data:
            points = [tuple(map(int, p)) for p in face['projected']] # Convert points to int tuples
            if len(points) >= 3: # Need at least 3 points to draw a polygon
                try:
                    pygame.draw.polygon(screen, face['color'], points)
                    # Draw black edges for definition
                    pygame.draw.polygon(screen, BLACK, points, 2) # 2 is line thickness
                except ValueError as e:
                    # Sometimes projection can result in invalid coords briefly during animation
                    # print(f"Warning: Skipping draw for invalid points: {points} ({e})")
                    pass

# --- Cube Class ---
class Cube:
    """ Manages the state, rendering, animation, and solving logic of the Rubik's Cube """
    def __init__(self):
        self.cubies = []
        self.reset() # Initialize cubies and state

        # Animation state
        self.animating = False
        self.animation_axis = None
        self.animation_angle = 0
        self.animation_target_angle = 0
        self.animation_speed = math.radians(18) # Degrees per frame (increased speed slightly)
        self.animation_cubies = []
        self.move_queue = deque()
        self.current_move_notation = None # Store notation like 'U', 'R''

        # Camera rotation state
        self.rotation_x = math.radians(25)
        self.rotation_y = math.radians(-35)
        self.camera_matrix = self.update_camera_matrix()

        # Mouse dragging state
        self.dragging = False
        self.last_mouse_pos = None

    def update_camera_matrix(self):
        """ Calculates the camera rotation matrix based on user rotation angles """
        rot_x = create_rotation_matrix(AXIS_X, self.rotation_x)
        rot_y = create_rotation_matrix(AXIS_Y, self.rotation_y)
        # Apply Y rotation first, then X for intuitive control
        return np.dot(rot_y, rot_x)

    def get_cubies_in_layer(self, axis_index, layer_coord):
        """ Returns cubies whose center position matches the layer coordinate along the axis """
        layer_cubies = []
        threshold = 0.1 # Use a small threshold for float comparison
        for cubie in self.cubies:
            if abs(cubie.pos[axis_index] - layer_coord) < threshold:
                layer_cubies.append(cubie)
        return layer_cubies

    def start_move(self, move_notation):
        """ Initiates a face rotation animation based on standard notation (e.g., 'U', "R'") """
        if self.animating:
            print("Warning: Cannot start move while animating.")
            return

        face_char = move_notation[0].upper()
        if face_char not in MOVE_DEFINITIONS:
            print(f"Error: Invalid move notation '{move_notation}'")
            return

        clockwise = (len(move_notation) == 1 or move_notation[1] != "'")

        axis_vec, angle_multiplier = MOVE_DEFINITIONS[face_char]
        # Layer coordinate is determined by the angle multiplier (e.g., U/D use Y-axis, R/L use X-axis)
        axis_index = np.argmax(np.abs(axis_vec)) # Find the dominant axis (0 for X, 1 for Y, 2 for Z)
        layer_coord = angle_multiplier # Layer coord is +1 or -1 based on face

        # Determine the actual rotation axis and angle
        # The rotation axis is the axis vector itself
        self.animation_axis = axis_vec
        # Angle depends on clockwise/counter-clockwise
        angle = math.pi / 2.0 # 90 degrees
        if not clockwise:
            angle = -angle
        # Apply the multiplier (handles faces like D, L, B rotating opposite to axis)
        self.animation_target_angle = angle * angle_multiplier

        # Find cubies in the layer to be rotated
        self.animation_cubies = self.get_cubies_in_layer(axis_index, layer_coord)
        if not self.animation_cubies:
             print(f"Warning: No cubies found for move {move_notation} (layer {layer_coord} on axis {axis_index})")
             return

        self.animating = True
        self.animation_angle = 0 # Reset current animation angle
        self.current_move_notation = move_notation # Store for logical update
        # print(f"Starting move: {move_notation}") # Debugging

    def queue_move(self, move_notation):
         """ Adds a move notation string to the queue """
         self.move_queue.append(move_notation)

    def start_next_queued_move(self):
        """ Starts the next move from the queue if available and not animating """
        if not self.animating and self.move_queue:
            move_notation = self.move_queue.popleft()
            self.start_move(move_notation)

    def update_animation(self):
        """ Updates the ongoing animation step by step """
        if not self.animating:
            self.start_next_queued_move() # Check queue if idle
            return

        # Calculate remaining angle and step size
        remaining_angle = self.animation_target_angle - self.animation_angle
        step = self.animation_speed * np.sign(remaining_angle) # Ensure step has correct sign

        # Prevent overshooting
        if abs(step) >= abs(remaining_angle):
            step = remaining_angle
            self.animating = False # Mark animation as finished for the next frame

        # Create rotation matrix for this small step
        step_rotation_matrix = create_rotation_matrix(self.animation_axis, step)

        # Rotate the cubies involved in the animation
        for cubie in self.animation_cubies:
            # Rotate around the global origin (0,0,0) as face centers align with axes
            cubie.rotate(step_rotation_matrix)

        # Update the total angle rotated in this animation
        self.animation_angle += step

        # If animation just finished, apply logical state update (optional here) and check queue
        if not self.animating:
            self.round_cubie_positions() # Snap positions after rotation
            # print(f"Finished move: {self.current_move_notation}") # Debugging
            self.current_move_notation = None
            self.start_next_queued_move() # Check queue for next move

    def round_cubie_positions(self):
        """ Rounds cubie positions and normals after animation to avoid floating point drift """
        for cubie in self.cubies:
            cubie.pos = np.round(cubie.pos)
            cubie.current_normals = np.round(cubie.current_normals)


    def scramble(self, num_moves=20):
        """ Queues a sequence of random moves """
        faces = list(MOVE_DEFINITIONS.keys())
        last_face = None
        for _ in range(num_moves):
            # Avoid redundant moves like R R' or R R R R
            possible_faces = [f for f in faces if f != last_face]
            face = random.choice(possible_faces)
            clockwise = random.choice([True, False])
            notation = face + ("" if clockwise else "'")
            self.queue_move(notation)
            last_face = face
        print(f"Queued {num_moves} scramble moves.")

    def reset(self):
        """ Resets the cube to its initial solved state """
        print("Resetting cube...")
        self.animating = False
        self.move_queue.clear()
        self.current_move_notation = None
        self.cubies = []
        # Create 26 cubies (ignore the center one at 0,0,0)
        for x in [-1, 0, 1]:
            for y in [-1, 0, 1]:
                for z in [-1, 0, 1]:
                    # Skip the invisible center piece
                    if abs(x) + abs(y) + abs(z) == 0:
                        continue
                    cubie = Cubie(x, y, z)
                    self.cubies.append(cubie)
        print("Cube Reset.")

    # --- Solver Logic (Beginner's Method - Partial Implementation) ---

    def find_cubie(self, criteria_func):
        """ Finds the first cubie matching the criteria function """
        for cubie in self.cubies:
            if criteria_func(cubie):
                return cubie
        return None

    def get_cubie_at(self, x, y, z):
         """ Gets the cubie at a specific logical coordinate """
         threshold = 0.1
         for cubie in self.cubies:
             if (abs(cubie.pos[0] - x) < threshold and
                 abs(cubie.pos[1] - y) < threshold and
                 abs(cubie.pos[2] - z) < threshold):
                 return cubie
         return None

    def get_face_color_at_normal(self, cubie, normal_vector):
        """ Gets the color of the face on a cubie pointing in the normal_vector direction """
        target_normal = tuple(map(int, np.round(normal_vector)))
        for i, normal in enumerate(np.round(cubie.current_normals)):
            if tuple(map(int, normal)) == target_normal:
                return cubie.face_colors[i]
        return None # Should not happen for a valid cubie face

    def solve_white_cross(self):
        """ Generates moves to solve the white cross on the top face (U face) """
        print("Solving White Cross...")
        solution_moves = []
        white_face_normal = np.array([0., 1., 0.]) # U face normal
        target_color = WHITE

        # Define target edge positions and their adjacent center colors
        # Format: (logical_pos, adjacent_face_normal, adjacent_color)
        targets = [
            ((0, 1, 1), np.array([0., 0., 1.]), FACE_COLORS['F']), # Front-Top edge
            ((1, 1, 0), np.array([1., 0., 0.]), FACE_COLORS['R']), # Right-Top edge
            ((0, 1, -1), np.array([0., 0., -1.]), FACE_COLORS['B']), # Back-Top edge
            ((-1, 1, 0), np.array([-1., 0., 0.]), FACE_COLORS['L']), # Left-Top edge
        ]

        edges_solved = 0
        max_iterations = 50 # Safety break
        iterations = 0

        while edges_solved < 4 and iterations < max_iterations:
            iterations += 1
            found_edge_in_cycle = False
            for target_pos, adj_normal, adj_color in targets:
                # Check if the target position is already solved
                current_cubie = self.get_cubie_at(*target_pos)
                if current_cubie:
                    white_color_on_top = self.get_face_color_at_normal(current_cubie, white_face_normal) == target_color
                    adj_color_correct = self.get_face_color_at_normal(current_cubie, adj_normal) == adj_color
                    if white_color_on_top and adj_color_correct:
                        continue # This edge is solved, move to the next target

                # --- Find the White/AdjColor Edge Piece ---
                # This requires searching all edge cubies (those with 2 colors)
                edge_piece = self.find_cubie(lambda c:
                    len([col for col in c.face_colors if col != BLACK]) == 2 and # Is edge piece
                    target_color in c.face_colors and adj_color in c.face_colors # Has the correct two colors
                )

                if not edge_piece:
                    print(f"Error: Could not find edge piece for {target_color} and {adj_color}")
                    continue # Should not happen on a valid cube

                # --- Move the Edge Piece to the Top Layer ---
                # This is a simplified approach. A full solver needs more robust logic
                # to handle pieces in middle/bottom layers or oriented incorrectly.

                # Example: If piece is on Front face (Z=1), not top (Y=1)
                current_pos = np.round(edge_piece.pos)
                if abs(current_pos[2] - 1) < 0.1 and abs(current_pos[1] - 0) < 0.1: # On F face middle layer edge
                   # Check orientation: Is white facing F or R/L?
                   colors = edge_piece.get_visible_face_colors()
                   white_dir = tuple(k for k, v in colors.items() if v == WHITE)[0]

                   # Simplified moves (needs refinement for general cases)
                   if white_dir == (0,0,1): # White facing Front
                       solution_moves.extend(["F", "U", "R", "U'"]) # Bring to top, placeholder
                   elif white_dir == (1,0,0) or white_dir == (-1,0,0): # White facing R or L
                       solution_moves.extend(["U"]) # Move top layer out of way, placeholder

                   found_edge_in_cycle = True
                   break # Re-evaluate after applying moves

                # Add more cases for other locations (Bottom layer, Middle layer sides, Top layer wrong slot/orientation)
                # ... (This part becomes very complex for a full solver) ...

            if found_edge_in_cycle:
                 # Apply the generated moves logically to update internal state for next iteration
                 # (In a real solver, you'd simulate these moves on an internal representation)
                 # For this demo, we just add to the queue and assume it will eventually work
                 pass # Moves are added to the main queue later
            else:
                 # If no edge was moved in a cycle, likely means remaining edges need complex moves
                 # or are already in the top layer but wrong place/orientation.
                 print("Solver (White Cross): Halting - requires more complex logic.")
                 break # Stop for this partial implementation

            # Check solved count (simplistic check)
            solved_count = 0
            for target_pos, adj_normal, adj_color in targets:
                 current_cubie = self.get_cubie_at(*target_pos)
                 if current_cubie:
                     white_ok = self.get_face_color_at_normal(current_cubie, white_face_normal) == target_color
                     adj_ok = self.get_face_color_at_normal(current_cubie, adj_normal) == adj_color
                     if white_ok and adj_ok:
                         solved_count += 1
            edges_solved = solved_count
            # print(f"Iteration {iterations}, Edges Solved: {edges_solved}") # Debugging

        if iterations >= max_iterations:
            print("Solver (White Cross): Reached max iterations.")

        # --- Queue the generated moves ---
        if solution_moves:
            print(f"Queuing {len(solution_moves)} moves for White Cross (Partial).")
            for move in solution_moves:
                self.queue_move(move)
        else:
            print("Solver (White Cross): No moves generated (or already solved/logic incomplete).")


    def solve(self):
        """ Initiates the solving process (currently only White Cross) """
        print("Attempting to solve...")
        # Clear any existing moves
        self.move_queue.clear()

        # --- Call Solver Stages ---
        # 1. White Cross
        self.solve_white_cross()

        # 2. White Corners (TODO)
        # self.solve_white_corners()

        # 3. Middle Layer Edges (TODO)
        # self.solve_middle_layer()

        # ... and so on for other layers ...

        if not self.move_queue:
             print("Solver: No moves generated.")


    def handle_input(self, event):
        """ Handles user input events for moves and camera """
        if event.type == pygame.KEYDOWN:
            key_map = {
                pygame.K_u: 'U', pygame.K_d: 'D',
                pygame.K_l: 'L', pygame.K_r: 'R',
                pygame.K_f: 'F', pygame.K_b: 'B',
            }
            if event.key in key_map:
                face = key_map[event.key]
                # Check for shift key for counter-clockwise moves (')
                mods = pygame.key.get_mods()
                clockwise = not (mods & pygame.KMOD_SHIFT)
                notation = face + ("" if clockwise else "'")
                self.queue_move(notation)

            elif event.key == pygame.K_s: # Scramble
                self.scramble()
            elif event.key == pygame.K_x: # Solve
                self.solve()
            elif event.key == pygame.K_c: # Reset Cube
                self.reset()

        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1: # Left mouse button
                self.dragging = True
                self.last_mouse_pos = event.pos
        elif event.type == pygame.MOUSEBUTTONUP:
            if event.button == 1:
                self.dragging = False
                self.last_mouse_pos = None
        elif event.type == pygame.MOUSEMOTION:
            if self.dragging and self.last_mouse_pos:
                mouse_pos = event.pos
                dx = mouse_pos[0] - self.last_mouse_pos[0]
                dy = mouse_pos[1] - self.last_mouse_pos[1]
                # Adjust rotation angles based on mouse movement
                sensitivity = 0.008
                self.rotation_y += dx * sensitivity
                self.rotation_x += dy * sensitivity
                # Clamp vertical rotation to avoid looking straight down/up
                self.rotation_x = max(-math.pi/2 + 0.1, min(math.pi/2 - 0.1, self.rotation_x))
                # Prevent excessive spinning
                self.rotation_y %= (2 * math.pi)

                self.camera_matrix = self.update_camera_matrix()
                self.last_mouse_pos = mouse_pos

    def draw(self, screen):
        """ Draws the background, the cube, and UI text """
        screen.fill(LIGHT_GRAY) # Background color

        # --- Draw Cubies ---
        # Sort cubies by their average Z distance from the camera in camera space
        # This helps with *inter-cubie* drawing order. Face sorting within cubies
        # handles *intra-cubie* drawing order.
        sorted_cubies = sorted(
            self.cubies,
            key=lambda c: np.dot(c.pos, self.camera_matrix.T)[2], # Z after camera rotation
            reverse=False # Draw closest cubies last (on top) - Correction: Farthest first is Painter's
        )
        sorted_cubies.reverse() # Draw farthest first

        # Draw each cubie
        for cubie in sorted_cubies:
            cubie.draw(screen, self.camera_matrix)

        # --- Draw UI Text ---
        try:
            font = pygame.font.Font(None, 24)
            controls_text = [
                "Controls:",
                "U/D/L/R/F/B: Rotate Face (Shift+Key for Prime)",
                "S: Scramble",
                "X: Solve (Partial - White Cross)",
                "C: Reset",
                "Mouse Drag: Rotate Camera",
                f"Moves Queued: {len(self.move_queue)}",
                f"Animating: {'Yes (' + self.current_move_notation + ')' if self.animating and self.current_move_notation else ('Yes' if self.animating else 'No')}"
            ]
            y_offset = 10
            for line in controls_text:
                text_surf = font.render(line, True, BLACK)
                screen.blit(text_surf, (10, y_offset))
                y_offset += 20
        except Exception as e:
            print(f"Error rendering font: {e}") # Handle font loading issues


# --- Main Game Class ---
class MainGame:
    """ Handles the main game loop, initialization, and cleanup """
    def __init__(self):
        try:
            pygame.init()
            # Attempt to initialize font module explicitly
            pygame.font.init()
            self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
            pygame.display.set_caption("Pygame 3D Rubik's Cube")
            self.clock = pygame.time.Clock()
            self.cube = Cube()
            print("Pygame initialized successfully.")
        except Exception as e:
            print(f"Error initializing Pygame: {e}")
            exit()

    def run(self):
        """ Main game loop """
        running = True
        while running:
            # Event handling
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                self.cube.handle_input(event)

            # Update cube state (process animations and queued moves)
            self.cube.update_animation()

            # Draw everything
            self.cube.draw(self.screen)

            # Update display
            pygame.display.flip()

            # Cap framerate
            self.clock.tick(60) # Aim for 60 FPS

        # Cleanup
        pygame.quit()
        print("Pygame closed.")

# --- Main Execution ---
if __name__ == '__main__':
    # Check for Pygame and NumPy (basic check)
    try:
        import pygame
        import numpy
    except ImportError:
        print("Error: Pygame and NumPy are required to run this program.")
        print("Please install them using: pip install pygame numpy")
        exit()

    print("Starting Pygame Rubik's Cube...")
    print("Controls:")
    print("  U, D, L, R, F, B keys: Rotate corresponding face clockwise")
    print("  Shift + (U, D, L, R, F, B): Rotate counter-clockwise")
    print("  S: Scramble the cube")
    print("  X: Trigger Solve function (Partial - attempts White Cross)")
    print("  C: Reset the cube to solved state")
    print("  Mouse Drag: Rotate the camera view")

    # Create and run the game
    game = MainGame()
    game.run()