from PyQt5.QtWidgets import QGraphicsScene

def shot_edit_delete_menu_function(scene, shot_data, current_si):
    def delete_coordinates(event):
        # Extract the clicked position
        x, y = event.scenePos().x(), event.scenePos().y()

        # Find the closest point in the shot data
        closest_index = None
        min_distance = float('inf')
        for i, (px, py) in enumerate(zip(shot_data['x'], shot_data['y'])):
            distance = (px - x)**2 + (py - y)**2
            if distance < min_distance:
                min_distance = distance
                closest_index = i

        if closest_index is not None:
            # Remove the point from the shot data
            shot_data['x'].pop(closest_index)
            shot_data['y'].pop(closest_index)
            shot_data['t'].pop(closest_index)

            # Update the scene
            scene.clear()
            for px, py in zip(shot_data['x'], shot_data['y']):
                scene.addEllipse(px - 2, py - 2, 4, 4)

            print(f"Deleted point at index {closest_index}.")

    # Connect the delete_coordinates function to the scene's mouse press event
    scene.mousePressEvent = delete_coordinates

    print("Shot edit/delete menu function initialized.")