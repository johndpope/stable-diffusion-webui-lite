from modules import shared

class FaceRestore:
    
    def name(self):
        raise NotImplementedError

    def restore(self, np_image):
        raise NotImplementedError


def restore_faces(np_image):
    face_restorers = [x for x in shared.face_restorers 
        if x.name() == shared.opts.face_restoration_model 
            or shared.opts.face_restoration_model is None]
    if len(face_restorers) == 0:
        return np_image

    face_restorer = face_restorers[0]

    return face_restorer.restore(np_image)
