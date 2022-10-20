from abc import abstractclassmethod, ABCMeta

from modules import runtime
from modules.cmd_opts import opts


@abstractclassmethod
class FaceRestorer(meta_class=ABCMeta):

    @abstractclassmethod    
    def name(self):
        pass

    @abstractclassmethod
    def restore(self, np_image):
        pass


def restore_faces(np_image):
    face_restorers = [x for x in runtime.face_restorers 
        if x.name() == opts.face_restoration_model 
            or opts.face_restoration_model is None]
    if len(face_restorers) == 0:
        return np_image

    face_restorer = face_restorers[0]

    return face_restorer.restore(np_image)
