import open3d as o3d
import numpy as np
import math
from PIL import Image
import pyvista as pv
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow.keras as keras
import matplotlib.pyplot as plt



# slide object is by:
# Image by <a href="https://www.freepik.com/3d-model/kids-slide-002_7582.htm#&position=0&from_view=search&uuid=aef06125-dc65-4e99-971c-0532239308e5">Freepik</a>


def create_data_base():
    mesh = pv.read("kids-slide-002-obj\kids-slide-002.obj")
    texture = pv.read_texture("kids-slide-002-obj\\textures\kids-slide-002-col-specular-4k.png")
    mesh.scale(1.0 / mesh.length, inplace=True)
    plotter = pv.Plotter(off_screen=True)
    plotter.add_mesh(mesh, texture=texture) 
    plotter.enable_lightkit()
    radius = mesh.length * 2

    for angle1 in range(0, 190, 10):
        for angle2 in range(0, 360, 10):
            x = radius * math.sin(math.radians(angle1)) * math.cos(math.radians(angle2))
            y = radius * math.sin(math.radians(angle1)) * math.sin(math.radians(angle2))
            z = radius * math.cos(math.radians(angle1))
            
            up_vector = (0,1,0)
            # #version 1 - at 0 and 180 angles, rotate the up vector
            # if(angle1 == 0):
            #     up_vector = (-math.cos(math.radians(angle2)), -math.sin(math.radians(angle2)), 0)
            # if(angle2 == 180):
            #     up_vector = (math.cos(math.radians(angle2)), math.sin(math.radians(angle2)), 0)

            # version 2 - at 0 and 180 angles, take only 1 screenshot (0, 0), (180, 0)
            if(angle1 == 0 and angle2 != 0):
                continue
              
            if(angle1 == 180 and angle2 != 0):
                continue
            

            plotter.camera_position = [(x, y, z), (0, 0, 0), up_vector]
            plotter.render() 
            # create a folder with the angles name if not exist
            if not os.path.exists(f"data_base\({angle1},{angle2})"):
                os.makedirs(f"data_base\({angle1},{angle2})")
            plotter.screenshot(f"data_base\({angle1},{angle2})\output_image_({angle1},{angle2}).png")

    for angle1 in np.arange(5, 20, 5):
        for angle2 in np.arange(0, 360, 5):
            x = radius * math.sin(math.radians(angle1)) * math.cos(math.radians(angle2))
            y = radius * math.sin(math.radians(angle1)) * math.sin(math.radians(angle2))
            z = radius * math.cos(math.radians(angle1))
            
            up_vector = (0,1,0)
            plotter.camera_position = [(x, y, z), (0, 0, 0), up_vector]
            plotter.render() 
            # create a folder with the angles name if not exist
            if not os.path.exists(f"data_base\({angle1},{angle2})"):
                os.makedirs(f"data_base\({angle1},{angle2})")
            plotter.screenshot(f"data_base\({angle1},{angle2})\output_image_({angle1},{angle2}).png")

    for angle1 in np.arange(80, 110, 5):
        for angle2 in np.arange(0, 360, 5):
            x = radius * math.sin(math.radians(angle1)) * math.cos(math.radians(angle2))
            y = radius * math.sin(math.radians(angle1)) * math.sin(math.radians(angle2))
            z = radius * math.cos(math.radians(angle1))
            
            up_vector = (0,1,0)
            plotter.camera_position = [(x, y, z), (0, 0, 0), up_vector]
            plotter.render() 
            # create a folder with the angles name if not exist
            if not os.path.exists(f"data_base\({angle1},{angle2})"):
                os.makedirs(f"data_base\({angle1},{angle2})")
            plotter.screenshot(f"data_base\({angle1},{angle2})\output_image_({angle1},{angle2}).png")

    for angle1 in np.arange(165, 180, 5):
        for angle2 in np.arange(0, 360, 5):
            x = radius * math.sin(math.radians(angle1)) * math.cos(math.radians(angle2))
            y = radius * math.sin(math.radians(angle1)) * math.sin(math.radians(angle2))
            z = radius * math.cos(math.radians(angle1))
            
            up_vector = (0,1,0)
            plotter.camera_position = [(x, y, z), (0, 0, 0), up_vector]
            plotter.render() 
            # create a folder with the angles name if not exist
            if not os.path.exists(f"data_base\({angle1},{angle2})"):
                os.makedirs(f"data_base\({angle1},{angle2})")
            plotter.screenshot(f"data_base\({angle1},{angle2})\output_image_({angle1},{angle2}).png")


    plotter.close() 
    return

def model_training():
    image_size = (48, 48)
    batch = 32

    train_datagen = ImageDataGenerator(
        rescale = 1./255,
        shear_range = 0.0,
        zoom_range = 0.0,
        horizontal_flip =False,
        vertical_flip =False,
        width_shift_range=0.0,  # No width shifting
        height_shift_range=0.0,  # No height shifting
        rotation_range=0  # No random rotation
    )

    val_datagen = ImageDataGenerator(rescale = 1./255)

    train_generator = train_datagen.flow_from_directory(
        'data_base/',
        target_size = (150,150),
        batch_size = 32,
        class_mode = 'categorical'
    )

    validation_generator = val_datagen.flow_from_directory(
        'data_base/',
        target_size = (150,150),
        batch_size = 32,
        class_mode = 'categorical'
    )

    model = Sequential()

    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)))
    model.add(MaxPooling2D((2, 2)))

    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))

    model.add(Flatten())

    model.add(Dense(128, activation='relu'))

    model.add(Dense(train_generator.num_classes, activation='softmax'))


    model.compile(
        loss ="categorical_crossentropy",
        optimizer = "adam",
        metrics = ["accuracy"]
    )

    model.fit(
        train_generator,
        epochs = 30,
        validation_data = validation_generator,

    )

    model.save("model.keras")


def testing():
    #load the mesh object
    mesh = pv.read("kids-slide-002-obj\kids-slide-002.obj")
    texture = pv.read_texture("kids-slide-002-obj\\textures\kids-slide-002-col-specular-4k.png")
    mesh.scale(1.0 / mesh.length, inplace=True)
    plotter = pv.Plotter(off_screen=True)
    plotter.add_mesh(mesh, texture=texture) 
    plotter.enable_lightkit()
    radius = mesh.length * 2

    #loading the keras model and train generator
    model = keras.models.load_model("model.keras")

    train_datagen = ImageDataGenerator(
        rescale = 1./255,
        shear_range = 0.0,
        zoom_range = 0.0,
        horizontal_flip =False,
        vertical_flip =False,
        width_shift_range=0.0,  # No width shifting
        height_shift_range=0.0,  # No height shifting
        rotation_range=0  # No random rotation
    )

    train_generator = train_datagen.flow_from_directory(
        'data_base/',
        target_size = (150,150),
        batch_size = 32,
        class_mode = 'categorical'
    )

    x_angles = []
    x_general = []
    y_list = []
    index = 0
    with open("testing_results.txt", "w") as file:
        file.write("actual angles\t|\tpredicted angles\t|\terror\n")
    # Rotate and take screenshots
    for angle1 in range(0, 185, 5):
        for angle2 in range(0, 360, 5):
            x = radius * math.sin(math.radians(angle1)) * math.cos(math.radians(angle2))
            y = radius * math.sin(math.radians(angle1)) * math.sin(math.radians(angle2))
            z = radius * math.cos(math.radians(angle1))
            
            
            up_vector = (0,1,0)
            # #version 1 - at 0 and 180 angles, rotate the up vector
            # if(angle1 == 0):
            #     up_vector = (-math.cos(math.radians(angle2)), -math.sin(math.radians(angle2)), 0)
            # if(angle2 == 180):
            #     up_vector = (math.cos(math.radians(angle2)), math.sin(math.radians(angle2)), 0)

            # version 2 - at 0 and 180 angles, take only 1 screenshot (0, 0), (180, 0)
            if angle1 == 0 and angle2 != 0:
                continue
            if angle1 == 180 and angle2 != 0:
                continue
            plotter.camera_position = [(x, y, z), (0, 0, 0), up_vector]
            plotter.render() 
            # create a folder with the angles name if not exist
            # if not os.path.exists(f"test_images\{angle1}_{angle2}"):
            #     os.makedirs(f"test_images\{angle1}_{angle2}")

            #save the image under the folder
            plotter.screenshot(f"test_images\output_image_({angle1},{angle2}).png")
            img1 = keras.utils.load_img(f"test_images\output_image_({angle1},{angle2}).png", target_size=(150,150))
            img_array1 = keras.utils.img_to_array(img1)
            img_array1 = keras.ops.expand_dims(img_array1, 0) 
            #img_array1 = img_array1 / 255.0

            predictions1 = model.predict(img_array1)
            result_index1 = np.argmax(predictions1)
            
            for k, v in train_generator.class_indices.items():
                if v == result_index1:
                    predicted_angles1 = k
            predictes_angles_num = list(map(float, predicted_angles1.strip("()").split(",")))
            error_angle1 = abs((predictes_angles_num[0]-angle1 + 180) % 360 - 180)
            error_angle2 = abs((predictes_angles_num[1]-angle2 + 180) % 360 - 180)
            y_list.append(error_angle1 + error_angle2)
            x_angles.append(angle1 + (angle2/100))
            x_general.append(index)

            index = index + 0.1
            error = f"({error_angle1},{error_angle2})"
            with open("testing_results.txt", "a") as file:
                file.write(f"({angle1}, {angle2})\t\t{predicted_angles1}\t\t{error}\n")
            # print(f"the image's actual angles: {angle1}_{angle2}")
            # print(f"the predicated index: {result_index1}")
            # print(f"the machine predicted angles: {predicted_angles1}")
    
    plotter.close() 
    #plot the results
    plt.plot(x_angles, y_list, marker='o', linestyle = '-')
    plt.xlabel("angles")
    plt.ylabel("error")
    plt.show()
    plt.plot(x_general, y_list, marker='o', linestyle = '-')
    plt.ylabel("error")
    plt.show()
    return

def angle_prediction():
    file_path = input("please anter image's path:")
    #loading the keras model and train generator
    model = keras.models.load_model("model.keras")

    train_datagen = ImageDataGenerator(
        rescale = 1./255,
        shear_range = 0.0,
        zoom_range = 0.0,
        horizontal_flip =False,
        vertical_flip =False,
        width_shift_range=0.0,  # No width shifting
        height_shift_range=0.0,  # No height shifting
        rotation_range=0  # No random rotation
    )

    train_generator = train_datagen.flow_from_directory(
        'data_base/',
        target_size = (150,150),
        batch_size = 32,
        class_mode = 'categorical'
    )
    img1 = keras.utils.load_img(file_path, target_size=(150,150))
    img_array1 = keras.utils.img_to_array(img1)
    img_array1 = keras.ops.expand_dims(img_array1, 0) 
    #img_array1 = img_array1 / 255.0

    predictions1 = model.predict(img_array1)
    result_index1 = np.argmax(predictions1)
    
    for k, v in train_generator.class_indices.items():
        if v == result_index1:
            predicted_angles1 = k
    print(f"predicted angles are: {predicted_angles1}")

def menu():
    while(True):
        num =input("choose from the list:\n1) create data set\n2) train the model\n3) test the model\n4) predicte angle\n5) quit\n")
        if num == "1":
            create_data_base()
        if num == "2":
            model_training()
        if num == "3":
            testing()
        if num == "4":
            angle_prediction()
        if num == "5":
            break



# create_data_base()
# model_training()
# testing()
# angle_prediction()
menu()

