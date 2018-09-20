FROM waleedka/modern-deep-learning:latest
MAINTAINER Yi Dong <yidong80@gmail.com>
# A MaskRCNN image based on image contained various Deep learning development with modern tools

# install MaskRCNN
# Get source from github
RUN git clone https://github.com/matterport/Mask_RCNN.git /usr/local/src/maskrcnn
# Dependencies
RUN pip3 --no-cache-dir install -r /usr/local/src/maskrcnn/requirements.txt
# Install
RUN cd /usr/local/src/maskrcnn/ && \
    python3 /usr/local/src/maskrcnn/setup.py install && \
    cd /
# Enivronment variables
ENV PYTHONPATH=/usr/local/src/maskrcnn/mrcnn:$PYTHONPATH

CMD ["/bin/bash"]