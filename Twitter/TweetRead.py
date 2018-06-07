from tweepy import OAuthHandler
from tweepy import Stream
from tweepy.streaming import StreamListener
import socket
import settings

# Set up your credentials
CONSUMER_KEY = 'Tvi6eL2N3rYqZTbNvxOyW6oQA'
CONSUMER_SECRET = 'ugSxqIlQWboyXWUBUKiGYbe47NjjaCNMDCh8V9HjREJV1DWjZ1'
ACCESS_TOKEN = '865325335876861952-UvCntYkF6MefGxdkBWwuQkQ43ahuktu'
ACCESS_SECRET = 'Yvk2dqJbQLPlBKKeRH5lyqhJ2iaPAt7GpcjaQaQTO3a6j'


class TweetsListener(StreamListener):

    def __init__(self, csocket):
        self.client_socket = csocket

    def on_data(self, data):
        try:
            #msg = json.loads(data)
            print(data)
            self.client_socket.send(data.encode('utf-8'))
            return True
        except BaseException as e:
            settings.socket_error_no = e
            print("Error on_data: %s" % str(e))
        return True

    def on_error(self, status):
        print(status)
        return True


def sendData(c_socket):
    auth = OAuthHandler(CONSUMER_KEY, CONSUMER_SECRET)
    auth.set_access_token(ACCESS_TOKEN, ACCESS_SECRET)
    twitter_stream = Stream(auth, TweetsListener(c_socket))
    twitter_stream.filter(track=["ama,ve,evet"], languages=['tr'])


if __name__ == "__main__":
    s = socket.socket()  # Create a socket object
    host = "localhost"  # Get local machine name
    port = 9995  # Reserve a port for your service.
    s.bind((host, port))  # Bind to the port
    print("Listening on port: %s" % str(port))
    s.listen(5)  # Now wait for client connection.
    c, addr = s.accept()  # Establish connection with client.
    print("Received request from: " + str(addr))
    sendData(c)
