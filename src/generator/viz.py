import matplotlib.pyplot as plt
import numpy as np

# usage: demo_linalg_norm(z_POS)
def demo_linalg_norm(z_POS, travel_delta_vector=[0.05, 0.10, 0.20, 0.40, 0.8], fig_ht=3):
    current_sum_POS= np.sum(z_POS, axis=0)
    current_n_POS = len(z_POS)
    current_mean_POS = current_sum_POS/current_n_POS
    fig,ax=plt.subplots(1,len(travel_delta_vector), figsize=(fig_ht*len(travel_delta_vector),fig_ht+1))
    for i, travel_delta in enumerate(travel_delta_vector):
        new_mean_POS = current_mean_POS - travel_delta
        travel=new_mean_POS-current_mean_POS
        x=[current_mean_POS[0:2][0],new_mean_POS[0:2][0],travel[0:2][0]]
        y=[current_mean_POS[0:2][1],new_mean_POS[0:2][1],travel[0:2][1]]
        for k,label in enumerate(['current','new','travel']):
            ax[i].plot([0,x[k]],[0,y[k]])
            ax[i].annotate('%s' % label, xy=[x[k],y[k]], textcoords='data')
        ax[i].set_title(f'travel factor={travel_delta}\nfrobenius norm={np.linalg.norm(travel):.2}')
    plt.suptitle("Demo of linal.norm's frobenius mode")
    fig.tight_layout()

#usage: vector_distribution(z_test)
def vector_distribution(z, alpha=0.005):
    for i in range(len(z)):
        plt.hist(z[i], alpha=alpha);
    plt.suptitle(f"z distribution, alpha={alpha}");

#usage: cursor=walk_images(cursor, next(data_flow), n_to_show)
def walk_images(cursor, example_batch, attribute_lists, n_to_show):
    from textwrap import wrap
    example_images = example_batch[0]
    fig = plt.figure(figsize=(15, 10))
    fig.subplots_adjust(hspace=0.4, wspace=0.4)
    for i in range(n_to_show):
        img = example_images[i].squeeze()
        sub = fig.add_subplot(2, n_to_show, i+1)
        sub.axis('off')
        sub.imshow(img)
        titlestr="\n".join(wrap(", ".join(attribute_lists[cursor]),30))

        sub.set_title(titlestr)
        cursor += 1
    plt.show()        
    return cursor

#usage: keypress_next_images(files_by_attribute, n_to_show)
# assumes df format like this:
# image_id,attr1<-1|1>, attr2<-1|1>, ...
def keypress_next_images(df, imageLoader, n_to_show, scale_value=255):
    '''
    If you shuffle the images, you can't get the labels back because there is more than one per image. So walk the dataset in order instead.
    '''
    data_flow = imageLoader.build(df=df,
                                  number_to_show=n_to_show, 
                                  scale_value = scale_value,
                                  shuffle=False)
    # get image attributes
    df=df.astype(str)
    for col_name in df.columns:
        df[col_name][df[col_name]=="-1"]=None
        df[col_name][df[col_name]=="1"]=col_name

    def listify(axis):
        # skip 1st column "image_id"
        return list(filter(None,list(axis[1:])))
    #listify(df.iloc[2,:]) # test it
    attribute_lists=df.iloc[:,1:].apply(listify,axis=1)

    stay=True
    cursor=0
    while stay:
        cursor = walk_images(cursor=cursor, 
                             example_batch=next(data_flow), 
                             attribute_lists=attribute_lists, 
                             n_to_show=n_to_show)
        key=input("Hit [enter] or enter 'c' to continue, enter any other key to quit [c]:")
        if( key != "c" and key !=""):
            stay=False

#usage: compare_images(n_to_show, example_images, reconst_images):
def compare_images(n_to_show, images, new_images):
    fig = plt.figure(figsize=(15, 3))
    fig.suptitle("Compare images: top=original, bottom=new")
    fig.subplots_adjust(hspace=0.4, wspace=0.4)
    for i in range(n_to_show):
        img = images[i].squeeze()
        sub = fig.add_subplot(2, n_to_show, i+1)
        sub.axis('off')        
        sub.imshow(img)
    for i in range(n_to_show):
        img = new_images[i].squeeze()
        sub = fig.add_subplot(2, n_to_show, i+n_to_show+1)
        sub.axis('off')
        sub.imshow(img)

def add_vector_to_images(feature_vec, example_batch, vae, label,
                         factors = [-4,-3,-2,-1,0,1,2,3,4],
                         n_to_show = 5
                         ):
    
    example_images = example_batch[0]
    example_labels = example_batch[1]

    _,_,z_points = vae.encoder.predict(example_images)


    fig = plt.figure(figsize=(18, 10))

    counter = 1

    for i in range(n_to_show):

        img = example_images[i].squeeze()
        sub = fig.add_subplot(n_to_show, len(factors) + 1, counter)
        sub.axis('off')        
        sub.imshow(img)

        counter += 1

        for factor in factors:

            changed_z_point = z_points[i] + feature_vec * factor
            changed_image = vae.decoder.predict(np.array([changed_z_point]))[0]

            img = changed_image.squeeze()
            sub = fig.add_subplot(n_to_show, len(factors) + 1, counter)
            sub.axis('off')
            sub.imshow(img)

            counter += 1
    plt.suptitle(f"Subtract-Add vector for {label}")
    plt.show()

def morph_faces(df, imageLoader, vae, start_image_file, end_image_file, factors = np.arange(0,1,0.1)):

    att_specific = df[df['image_id'].isin([start_image_file, end_image_file])]
    att_specific = att_specific.reset_index()
    data_flow_label = imageLoader.build(att_specific, 2)

    example_batch = next(data_flow_label)
    example_images = example_batch[0]
    example_labels = example_batch[1]

    print("Predicting latent vectors for {len(example_images)} images...")
    _,_,z_points = vae.encoder.predict(example_images,verbose=0)

    fig = plt.figure(figsize=(18, 8))

    counter = 1

    img = example_images[0].squeeze()
    sub = fig.add_subplot(1, len(factors)+2, counter)
    sub.axis('off')        
    sub.imshow(img)

    counter+=1

    for factor in factors:

        changed_z_point = z_points[0] * (1-factor) + z_points[1]  * factor
        changed_image = vae.decoder.predict(np.array([changed_z_point]))[0]

        img = changed_image.squeeze()
        sub = fig.add_subplot(1, len(factors)+2, counter)
        sub.axis('off')
        sub.imshow(img)

        counter += 1

    img = example_images[1].squeeze()
    sub = fig.add_subplot(1, len(factors)+2, counter)
    sub.axis('off')        
    sub.imshow(img)

    fig.suptitle(f"Morphed {start_image_file} to {end_image_file}")
    plt.show()
