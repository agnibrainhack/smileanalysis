import api.face_detectionHaarCascade as hrcd
class Image:
    def __init__(self,url):
        self.url = url

    def ret(self):
        answer = hrcd.detect(self.url)
        return answer
