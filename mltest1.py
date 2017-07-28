import numpy as np

class team:
    games = 10
    roster = {}
    def __init__(self):
        self.players = 10
        self.name = "KT"

    #determine length of schedule
    def sched(self, players):
        return players*self.games
    
    #set captains
    def captain(self, title, name):
        self.roster[title] = name

def main():
    print("working")
    teamName = team()

    print(teamName.players)
    print(teamName.sched(teamName.players))
    teamName.captain("co-captain1", "James")
    teamName.captain("co-captain2", "Stephen")
    print(teamName.roster["co-captain1"])
    print(teamName.roster["co-captain2"])

if __name__ == "__main__":
    main()
