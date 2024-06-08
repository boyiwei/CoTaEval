import sys
import time

def num_to_bytes(some_int):
    return str(some_int).encode()

def generate_redis_protocol_basic(*cmd):
    """generate_redis_protocol_basic. Returns a single redis command byte string. 

    :param cmd:
    """

    arg_byte = b"$"
    array_byte = b"*"
    empty_byte = b""
    line_end_bytes = b"\r\n"
    proto = empty_byte.join((array_byte, num_to_bytes(len(cmd)), line_end_bytes))
    # linear builder pattern

    argument_buffer = [proto]
    for elt in cmd:
        if type(elt) is not bytes:
            elt = elt.encode()
        argument_buffer.append(empty_byte.join((arg_byte, num_to_bytes(len(elt)), line_end_bytes, elt, line_end_bytes)))
    proto = empty_byte.join(argument_buffer)

    return proto

def print_b(some_bytes):
    sys.stdout.buffer.write(some_bytes)

# buffer_view = memoryview(bytearray(1024))

# def generate_redis_buffered(*cmd):

    # arg_byte = b"$"
    # array_byte = b"*"
    # empty_byte = b""
    # line_end = b"\r\n"

    # cursor = 0

    # def b_write(buffer, data, idx):
        # l = len(data)
        # buffer[idx:idx+l] = data
        # return cursor + l
    
    # # array header
    # cursor = b_write(buffer_view, array_byte, cursor)
    # cursor = b_write(buffer_view, num_to_bytes(len(cmd)), cursor)
    # cursor = b_write(buffer_view, line_end, cursor)
    
    # for elt in cmd:
        # elt = elt.encode()

        # # write the length of the element
        # cursor = b_write(buffer_view, arg_byte, cursor)
        # cursor = b_write(buffer_view, num_to_bytes(len(elt)), cursor)
        # cursor = b_write(buffer_view, line_end, cursor)

        # # write the full element
        # cursor = b_write(buffer_view, elt, cursor)
        # cursor = b_write(buffer_view, line_end, cursor)

    # return buffer_view[:cursor]


# s = generate_redis_buffered("PFADD", "hll.bench", "10")
# print_b(s)
if __name__ == '__main__':

    N = 1000000
    xs = [str(i) for i in range(N)]

    start = time.time()
    for x in xs:
        print_b(generate_redis_protocol_basic("PFADD", "hll.bench", x))
        # print_b(generate_redis_buffered("PFADD", "hll.bench", x))
    end = time.time()

    print((end - start) / N, file=sys.stderr)
    print(end - start, file=sys.stderr)
