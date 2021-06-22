#include "Clustering.h"

#define CARLA_DATA 0
#define LYFT_DATA 1

/// Driver function to sort the vector elements by first element of pair in descending order
bool sortbysecrev(const std::pair<float,int> &a, const std::pair<float,int> &b)
{
    return (a.first > b.first);
}

void Clustering::objects_clustering(VecArray &points, VecArray &objectsAboveRoad, calibMat& calib, vec& egoLocation, cv::Mat& img, pcl::PointCloud<PointTypeIO>::Ptr cloud_out, pcl::IndicesClustersPtr clusters) {
    // Data containers used
    pcl::PointCloud<PointTypeIO>::Ptr cloud_in (new pcl::PointCloud<PointTypeIO>);
    pcl::PointCloud<PointTypeFull>::Ptr cloud_with_normals (new pcl::PointCloud<PointTypeFull>);
    pcl::IndicesClustersPtr small_clusters (new pcl::IndicesClusters), large_clusters (new pcl::IndicesClusters);
    pcl::search::KdTree<PointTypeIO>::Ptr search_tree (new pcl::search::KdTree<PointTypeIO>);
    pcl::console::TicToc tt;

    // Load the input point cloud
    cloud_in->points.clear();
    /// resize space for point cloud
    cloud_in->width = points.size(); cloud_in->height = 1;
    cloud_in->points.resize(cloud_in->width * cloud_in->height);
    int indPoint = 0;
    for (auto & point : cloud_in->points) {
        point.x = points[indPoint].x;
        point.y = points[indPoint].y;
        point.z = points[indPoint].z;
        indPoint++;
    }

    // Downsample the cloud using a Voxel Grid class
    std::cerr << "Downsampling...\n", tt.tic ();
    pcl::VoxelGrid<PointTypeIO> vg;
    vg.setInputCloud (cloud_in);
    vg.setLeafSize (0.01, 0.01, 0.01);
    vg.setDownsampleAllData (true);
    vg.filter (*cloud_out);
    std::cerr << ">> Done: " << tt.toc () << " ms, " << cloud_out->points.size () << " points\n";

    // Set up a Normal Estimation class and merge data in cloud_with_normals
    std::cerr << "Computing normals...\n", tt.tic ();
    pcl::copyPointCloud (*cloud_out, *cloud_with_normals);
    pcl::NormalEstimation<PointTypeIO, PointTypeFull> ne;
    ne.setInputCloud (cloud_out);
    ne.setSearchMethod (search_tree);
    ne.setRadiusSearch (10.0);
    ne.compute (*cloud_with_normals);
    std::cerr << ">> Done: " << tt.toc () << " ms\n";

    // Set up a Conditional Euclidean Clustering class
    std::cerr << "Segmenting to clusters...\n", tt.tic ();
    pcl::ConditionalEuclideanClustering<PointTypeFull> cec (true);
    cec.setInputCloud (cloud_with_normals);
    cec.setConditionFunction (&customRegionGrowing);

    #if CARLA_DATA == 1
        cec.setClusterTolerance (0.8);
        cec.setMinClusterSize (10);
        cec.setMaxClusterSize (10000);
    #endif

    #if LYFT_DATA == 1
        cec.setClusterTolerance (0.8);
        cec.setMinClusterSize (10);
        cec.setMaxClusterSize (10000);
    #endif

    cec.segment (*clusters);
    cec.getRemovedClusters (small_clusters, large_clusters);
    std::cerr << ">> Done: " << tt.toc () << " ms\n";

    // Using the intensity channel for lazy visualization of the output
    for (int i = 0; i < small_clusters->size (); ++i)
        for (int j = 0; j < (*small_clusters)[i].indices.size (); ++j)
            cloud_out->points[(*small_clusters)[i].indices[j]].intensity = -2.0;
    for (int i = 0; i < large_clusters->size (); ++i)
        for (int j = 0; j < (*large_clusters)[i].indices.size (); ++j)
            cloud_out->points[(*large_clusters)[i].indices[j]].intensity = +10.0;

    for (int i = 0; i < clusters->size (); ++i)
    {
        int label = rand () % 8;
        for (int j = 0; j < (*clusters)[i].indices.size (); ++j)
            cloud_out->points[(*clusters)[i].indices[j]].intensity = label;
    }
}
void Clustering::curbs_clustering(VecArray &points, VecArray &debug) {
    /// increase iter by 1
    m_iter++;

    /// Data containers used
    pcl::PointCloud<PointTypeIO>::Ptr cloud_in (new pcl::PointCloud<PointTypeIO>), cloud_out (new pcl::PointCloud<PointTypeIO>);
    pcl::PointCloud<PointTypeFull>::Ptr cloud_with_normals (new pcl::PointCloud<PointTypeFull>);
    pcl::IndicesClustersPtr clusters (new pcl::IndicesClusters), small_clusters (new pcl::IndicesClusters), large_clusters (new pcl::IndicesClusters);
    pcl::search::KdTree<PointTypeIO>::Ptr search_tree (new pcl::search::KdTree<PointTypeIO>);
    pcl::console::TicToc tt;

    /// Load the input point cloud
    cloud_in->points.clear();
    /// resize space for point cloud
    cloud_in->width = points.size(); cloud_in->height = 1;
    cloud_in->points.resize(cloud_in->width * cloud_in->height);
    int indPoint = 0;
    for (auto & point : cloud_in->points) {
        point.x = points[indPoint].x;
        point.y = points[indPoint].y;
        point.z = points[indPoint].z;
        indPoint++;

    }

    /// Downsample the cloud using a Voxel Grid class
    std::cerr << "Downsampling...\n", tt.tic ();
    pcl::VoxelGrid<PointTypeIO> vg;
    vg.setInputCloud (cloud_in);
    vg.setLeafSize (0.1, 0.1, 0.1);
    vg.setDownsampleAllData (true);
    vg.filter (*cloud_out);
    std::cerr << ">> Done: " << tt.toc () << " ms, " << cloud_out->points.size () << " points\n";

    /// Set up a Normal Estimation class and merge data in cloud_with_normals
    std::cerr << "Computing normals...\n", tt.tic ();
    pcl::copyPointCloud (*cloud_out, *cloud_with_normals);
    pcl::NormalEstimation<PointTypeIO, PointTypeFull> ne;
    ne.setInputCloud (cloud_out);
    ne.setSearchMethod (search_tree);

    #if CARLA_DATA == 1
        ne.setRadiusSearch (2.0); /// old
    #endif

    #if LYFT_DATA == 1
        ne.setRadiusSearch (5.0); /// old
    #endif

    ne.compute (*cloud_with_normals);
    std::cerr << ">> Done: " << tt.toc () << " ms\n";

    /// Set up a Conditional Euclidean Clustering class
    std::cerr << "Segmenting to clusters...\n", tt.tic ();
    pcl::ConditionalEuclideanClustering<PointTypeFull> cec (true);
    cec.setInputCloud (cloud_with_normals);
    cec.setConditionFunction (&customRegionGrowing);

    #if CARLA_DATA == 1
        cec.setClusterTolerance (2.0); /// old


//        cec.setClusterTolerance (3.0); /// test


        cec.setMinClusterSize (5);
    #endif

    #if LYFT_DATA == 1
        cec.setClusterTolerance (4.0);
        cec.setMinClusterSize (3);
    #endif

    cec.setMaxClusterSize (1000);
    cec.segment (*clusters);
    cec.getRemovedClusters (small_clusters, large_clusters);
    std::cerr << ">> Done: " << tt.toc () << " ms\n";

    /// Using the intensity channel for lazy visualization of the output
    for (int i = 0; i < small_clusters->size (); ++i)
        for (int j = 0; j < (*small_clusters)[i].indices.size (); ++j)
            cloud_out->points[(*small_clusters)[i].indices[j]].intensity = -2.0;

    for (int i = 0; i < large_clusters->size (); ++i)
        for (int j = 0; j < (*large_clusters)[i].indices.size (); ++j)
            cloud_out->points[(*large_clusters)[i].indices[j]].intensity = +10.0;

    /// TODO : choose the clusters that are closer to the last clusters...

//    std::cout << " clusters = " << clusters->size() << std::endl;

    if (clusters->size() <= 1) {
        m_clusters = 1;
        return;
    }
    else
        m_clusters =  clusters->size();

    m_iter = 1; /// to remove it or not!!

    /// if first time then choose the two biggest clusters
    if (m_iter == 1) {
        pairFI clustersLenId;
        for (unsigned int l=0; l<clusters->size(); l++) {
            clustersLenId.emplace_back(std::make_pair((*clusters)[l].indices.size(), l));
        }
        sort(clustersLenId.begin(), clustersLenId.end(), sortbysecrev); /// sort clustersLenId in decreasing order of first element

        /// store points of the first biggest cluster
        int clusterId = clustersLenId[0].second;
        points.clear();
        curbUp.clear();
        vec p, mean_point{0.0f, 0.0f, 0.0f};
        for (int j = 0; j < (*clusters)[clusterId].indices.size (); ++j) {
            p = vec(cloud_out->points[(*clusters)[clusterId].indices[j]].x,
                    cloud_out->points[(*clusters)[clusterId].indices[j]].y,
                    cloud_out->points[(*clusters)[clusterId].indices[j]].z);
            points.emplace_back(p);

            mean_point.x += p.x;
            mean_point.y += p.y;
            mean_point.z += p.z;

            curbUp.emplace_back(p);
            debug.emplace_back(p);
            debug.emplace_back(vec(0.0f, 0.0f, 1.0f));
        }
        if ((*clusters)[clusterId].indices.size () > 0)
            mean_point /= (*clusters)[clusterId].indices.size ();
        lastCurbPoints.emplace_back(mean_point);

        mean_point.x = mean_point.y = mean_point.z = 0.0f;

        curbBelow.clear();
        /// store points of the second biggest cluster
        clusterId = clustersLenId[1].second;
        for (int j = 0; j < (*clusters)[clusterId].indices.size (); ++j) {

            p = vec(cloud_out->points[(*clusters)[clusterId].indices[j]].x,
                    cloud_out->points[(*clusters)[clusterId].indices[j]].y,
                    cloud_out->points[(*clusters)[clusterId].indices[j]].z);
            points.emplace_back(p);

            mean_point.x += p.x;
            mean_point.y += p.y;
            mean_point.z += p.z;

            curbBelow.emplace_back(p);
            debug.emplace_back(p);
            debug.emplace_back(vec(0.0f, 1.0f, 0.0f));
        }

        if ((*clusters)[clusterId].indices.size () > 0)
            mean_point /= (*clusters)[clusterId].indices.size ();
        lastCurbPoints.emplace_back(mean_point);
    }
}
