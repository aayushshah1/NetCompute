# import socket

# ip = "10.125.1.41"
# open_ports = []

# for port in range(5000, 6001):
#     with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
#         sock.settimeout(0.5)
#         result = sock.connect_ex((ip, port))
#         if result == 0:
#             print(f"Port {port} is OPEN")
#             open_ports.append(port)

# print("✅ Use these ports:", open_ports[:4])

import socket

def find_available_ports(start=5000, end=6000, count=4):
    available_ports = []
    for port in range(start, end + 1):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("", port))  # Bind to all interfaces
                available_ports.append(port)
                if len(available_ports) == count:
                    break
            except:
                continue
    return available_ports

ports = find_available_ports()
print("✅ Available ports to use:", ports)