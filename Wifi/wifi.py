import socket

#UDP binding port
UDP_PORT = 8086


class WifiSocket:
    def __int__(self, sock = None):

        if sock is None:
            # Create a UDP socket
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        else:
            self.sock = sock
        # Bind the socket to the port
        server_address = ('', UDP_PORT)
        self.sock.bind(server_address)

    # Receive data from Wifi
    def wifiReceive(self, revNumber = 128):
        data, address = self.sock.recvfrom(revNumber)
        return data, address[0]

    # Send data from Wifi
    def wifiSend(self, data, ip):
        if ip:
            self.sock.sendto(data, (ip, UDP_PORT))


if __name__ =='__main__':

    # Create an WifiSocket instance
    wifi = WifiSocket()

    # Initialize the WifiSocket
    wifi.__int__()

    # Test wifi send and receive
    while True:
        data = wifi.wifiReceive()

