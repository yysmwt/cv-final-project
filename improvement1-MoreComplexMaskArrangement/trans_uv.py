import random

from PIL import Image
import numpy as np
from matplotlib import pyplot as plt

N = 4
block = 64
iter = 0
def load_image(path):
    image = Image.open(path)
    # 转换为 RGB 格式
    rgb_image = image.convert('RGB')
    return np.array(rgb_image)

def swap_cube(uv,gx,gy,fx,fy,size):
    global iter
    if abs(gx - fx) < size and abs(gy - fy) < size:
        return
    if gx + size > N or gy + size >  N or fx + size > N or fy + size > N:
        return
    iter += 1
    temp = np.copy(uv[gy * block:(gy + size) * block, gx * block:(gx + size) * block])
    uv[gy * block:(gy + size) * block, gx * block:(gx + size) * block] = np.copy(uv[fy * block:(fy + size) * block, fx * block:(fx + size) * block])
    uv[fy * block:(fy + size) * block, fx * block:(fx + size) * block] = temp
    if random.choice([True, False]):
        uv[gy * block:(gy + size) * block, gx * block:(gx + size) * block] = np.transpose(uv[gy * block:(gy + size) * block, gx * block:(gx + size) * block],(1,0,2))
    if random.choice([True, False]):
        uv[fy * block:(fy + size) * block, fx * block:(fx + size) * block] = np.transpose(uv[fy * block:(fy + size) * block, fx * block:(fx + size) * block],(1,0,2))

def swap_triangle(uv,gx,gy,fx,fy,size):
    global iter
    if random.choice([True,False]):
        if abs(gx - fx) < size and abs(gy - fy) < size:
            return
        if gy - size < 0 or gx - size < 0 or gx + size > N or fy + size > N or fx - size < 0 or fx + size > N:
            return
        else:
            iter += 1
            for i in range(size * block):
                gy_1 = gy * block - i - 1
                fy_1 = fy * block + i
                temp = np.copy(uv[gy_1:gy_1 + 1,gx * block - i:gx * block + i + 1,:])
                print(gy_1,fy_1)
                uv[gy_1:gy_1 + 1,gx * block - i:gx * block + i + 1,:] = np.copy(uv[fy_1:fy_1 + 1,fx * block - i:fx * block + i + 1,:])
                uv[fy_1:fy_1 + 1,fx * block - i:fx * block + i + 1,:] = temp

    else:
        if gy == fy and abs(gx - fx) < 2 * size:
            return
        if abs(gx - fx) < size and abs(gy - fy) < size:
            return
        if gy - size < 0 or gx - size < 0 or gx + size > N or fy - size < 0 or fx - size < 0 or fx + size > N:
            return
        else:
            iter += 1
            for i in range(size * block):
                gy_1 = gy * block - i - 1
                fy_1 = fy * block - i - 1
                temp = np.copy(uv[gy_1:gy_1 + 1,gx * block - i:gx * block + i + 1,:])
                uv[gy_1:gy_1 + 1,gx * block - i:gx * block + i + 1,:] = np.copy(uv[fy_1:fy_1 + 1,fx * block - i:fx * block + i + 1,:])
                uv[fy_1:fy_1 + 1,fx * block - i:fx * block + i + 1,:] = temp
def swap_triangle_row(uv,gx,gy,fx,fy,size):
    global iter
    if random.choice([True,False]):
        if gx - size < 0 or gy - size < 0 or gy + size > N or fx + size > N or fy - size < 0 or fy + size > N:
            return
        else:
            iter += 1
            for i in range(size * block):
                gx_1 = gx * block - i - 1
                fx_1 = fx * block + i
                temp = np.copy(uv[gy * block - i:gy * block + i + 1,gx_1:gx_1 + 1,:])
                uv[gy * block - i:gy * block + i + 1,gx_1:gx_1 + 1,:] = np.copy(uv[fy * block - i:fy * block + i + 1,fx_1:fx_1 + 1,:])
                print(gx_1,fx_1)
                uv[fy * block - i:fy * block + i + 1,fx_1:fx_1 + 1,:] = temp

    else:
        if gx == fx and abs(fy - fy) < 2 * size:
            return
        if abs(gx - fx) < size and abs(gy - fy) < size:
            return
        if gx - size < 0 or gy - size < 0 or gy + size > N or fx - size < 0 or fy - size < 0 or fy + size > N:
            return
        else:
            iter += 1
            for i in range(size * block):
                gx_1 = gx * block - i - 1
                fx_1 = fx * block - i - 1
                temp = np.copy(uv[gy * block - i:gy * block + i + 1,gx_1:gx_1 + 1,:])
                print(gx_1, fx_1)
                uv[gy * block - i:gy * block + i + 1,gx_1:gx_1 + 1,:] = np.copy(uv[fy * block - i:fy * block + i + 1,fx_1:fx_1 + 1,:])
                uv[fy * block - i:fy * block + i + 1,fx_1:fx_1 + 1,:] = temp

def trans_uv(trans:int=10,unit:int=4):
    """

    :param trans:how many trans function will do
    :param unit: the min length of polygon, default is 4 means 1/4 photo size
    不妨先分几个情况分别决定要对什么样的进行替换就好了，

    :return:
    """
    global iter
    uv = load_image('improvement1-mask/uv_map_a.png')

    while iter < 2:
        gx = random.randint(0, N - 1)
        gy = random.randint(0, N - 1)
        fx = random.randint(0, N - 1)
        fy = random.randint(0, N - 1)
        size = random.random()
        if size < 0.1:
            size = 1
        else:
            size = 2
        swap_cube(uv,gx,gy,fx,fy,size)
    iter = 0
    while iter < 2:

        gx = random.randint(0, N)
        gy = random.randint(0, N)
        fx = random.randint(0, N)
        fy = random.randint(0, N)
        size = random.random()
        if size < 0.1:
            size = 1
        else:
            size = 2
        swap_triangle(uv,gx,gy,fx,fy,size)
    iter = 0
    while iter < 2:
        gx = random.randint(0, N)
        gy = random.randint(0, N)
        fx = random.randint(0, N)
        fy = random.randint(0, N)
        size = random.random()
        if size < 0.1:
            size = 1
        else:
            size = 2
        swap_triangle_row(uv,gx,gy,fx,fy,size)

    plt.imshow(uv)
    plt.show()
    image = Image.fromarray(uv)

    image.save('trans_uv.png')
    print('done')

trans_uv()
