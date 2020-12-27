
import numpy as np

import pyglet

problem = np.array([[0, 7, 0, 6, 1, 0, 0, 2, 0],
       [6, 1, 4, 9, 2, 0, 3, 7, 5],
       [9, 0, 8, 3, 7, 5, 0, 0, 0],
       [1, 0, 0, 0, 4, 0, 0, 0, 0],
       [0, 4, 6, 0, 0, 9, 0, 0, 0],
       [0, 8, 9, 0, 0, 3, 2, 4, 6],
       [5, 0, 7, 4, 3, 0, 8, 6, 0],
       [4, 0, 1, 8, 0, 0, 0, 9, 0],
       [0, 6, 0, 5, 0, 7, 0, 0, 0]])
dim = 3


screen = pyglet.canvas.get_display().get_default_screen()
window_width = int(min(screen.width, screen.height) * 2 / 3)
window_height = int(window_width * 1.1)
window = pyglet.window.Window(window_width, window_height)

config = {2: (36, 60, 40),
          3: (24, 27.5, 15),
          4: (14, 15, 6.675),
          5: (8, 9, 6.675)}

cursor = window.get_system_mouse_cursor(window.CURSOR_CROSSHAIR)
window.set_mouse_cursor(cursor)

@window.event
def on_draw():
    
    pyglet.gl.glClearColor(1,1,1,1)


    label = pyglet.text.Label('S U D O K U', font_name='Garamond', font_size=36, x=window_width/2, y=window_height - 20,
                  anchor_x='center', anchor_y='top', color=(0, 0, 0, 192), width=window_width / 2,
                  align='center', multiline=True)
    label.draw()
    
    batch = pyglet.graphics.Batch()
    pyglet.gl.glLineWidth(1)
    board_size = dim**2 + 1
    lower_grid_coord = 50
    upper_grid_coord = 525

    delta = (upper_grid_coord - lower_grid_coord)/(board_size - 1)
    left_coord = lower_grid_coord
    right_coord = lower_grid_coord
    ver_list = []
    color_list = []
    num_vert = 0
    for i in range(board_size):
        # horizontal
        ver_list.extend((lower_grid_coord, left_coord,
                         upper_grid_coord, right_coord))
        # vertical
        ver_list.extend((left_coord, lower_grid_coord,
                         right_coord, upper_grid_coord))
        color_list.extend([0.3, 0.3, 0.3] * 4)  # black
        left_coord += delta
        right_coord += delta
        num_vert += 4
    
    batch.add(num_vert, pyglet.gl.GL_LINES, None,
              ('v2f/static', ver_list), ('c3f/static', color_list))

    batch.draw()
    
    batch = pyglet.graphics.Batch()
    pyglet.gl.glLineWidth(3)
    board_size = dim + 1
    lower_grid_coord = 50
    upper_grid_coord = 525
    delta = (upper_grid_coord - lower_grid_coord)/(board_size - 1)
    left_coord = lower_grid_coord
    right_coord = lower_grid_coord
    ver_list = []
    color_list = []
    num_vert = 0
    for i in range(board_size):
        # horizontal
        ver_list.extend((lower_grid_coord, left_coord,
                         upper_grid_coord, right_coord))
        # vertical
        ver_list.extend((left_coord, lower_grid_coord,
                         right_coord, upper_grid_coord))
        color_list.extend([0.3, 0.3, 0.3] * 4)  # black
        left_coord += delta
        right_coord += delta
        num_vert += 4
    
    batch.add(num_vert, pyglet.gl.GL_LINES, None,
              ('v2f/static', ver_list), ('c3f/static', color_list))
    
    batch.draw()

    
    batch = pyglet.graphics.Batch()
    font_size, horizontal_shift, vertical_shift = config[dim]
    lower_grid_coord = 50
    upper_grid_coord = 525
    delta = (upper_grid_coord - lower_grid_coord)/(dim**2)
    vertical_coord = lower_grid_coord
    horizontal_coord = lower_grid_coord
    for i, row in enumerate(np.rot90(problem, -1).tolist()):
        for value in row:
            if value:
                pyglet.text.Label(str(value),
                                  font_name='Arial',
                                  font_size=font_size,
                                  x=horizontal_coord + horizontal_shift,
                                  y=vertical_coord + vertical_shift,
                                  anchor_x='center',
                                  color=(0, 0, 0, 192), batch=batch)
            vertical_coord += delta
        vertical_coord = lower_grid_coord
        horizontal_coord += delta
        
    batch.draw()      


pyglet.app.run()
