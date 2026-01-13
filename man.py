class Man:
    def __init__(self, name):
        self.name = name
        print("Initialized Finished")
    
    def hello(self):
        print("Hello"+ self.name + "!")

    def goodbye(self, type="English"):
        if type == "English":
            print("goodbye" + self.name)
        else: 
            print("bye bye")

m = Man("David")
m.goodbye("Chinese")