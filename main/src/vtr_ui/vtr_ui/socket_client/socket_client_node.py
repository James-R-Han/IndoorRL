#!/usr/bin/env python3

from vtr_mission_planning.mission_client_builder import build_master_client
from .socket_client import SocketMissionClient


def main():

  client, mgr = build_master_client(SocketMissionClient)

  # This has to happen in a specific order:

  # 1) Start the mission client, spawning a separate process with ROS in it
  client.start()

  # 2) Start a python Multiprocessing.Manager that contains the client.  This blocks the main process.
  mgr.get_server().serve_forever()

  # 3) The server is interrupted (Ctrl-C), which shuts down the Multiprocessing.Manager
  # 4) Shut down the client, stopping ROS and joining the process that was spawned in (1)
  client.shutdown()

  # 5) Tell the web server to exit using a SocketIO request, as it cannot be killed cleanly from the terminal
  client.kill_server()


if __name__ == '__main__':
  main()
