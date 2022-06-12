/**
 * Creating a sidebar enables you to:
 - create an ordered group of docs
 - render a sidebar for each doc of that group
 - provide next/previous navigation

 The sidebars can be generated from the filesystem, or explicitly defined here.

 Create as many sidebars as you want.
 */

// @ts-check

/** @type {import('@docusaurus/plugin-content-docs').SidebarsConfig} */
const sidebars = {
  docs: [
    {
      type: 'doc',
      id: 'introduction',
    },
    {
      type: 'category',
      collapsible: true,
      collapsed: false,
      label: 'Feature',
      items: [
        {
          type: 'category',
          collapsible: true,
          collapsed: false,
          label: 'Geometric Distortion',
          items: [
            'feature/geometric-distortion/interface',
            'feature/geometric-distortion/example',
            {
              type: 'category',
              collapsible: true,
              collapsed: false,
              label: 'Implementation',
              items: [
                'feature/geometric-distortion/camera',
                'feature/geometric-distortion/mls',
                'feature/geometric-distortion/affine',
              ]
            },
          ]
        },
        {
          type: 'category',
          collapsible: true,
          collapsed: false,
          label: 'Photometric Distortion',
          items: [
            'feature/photometric-distortion/interface',
            'feature/photometric-distortion/example',
            {
              type: 'category',
              collapsible: true,
              collapsed: false,
              label: 'Implementation',
              items: [
                'feature/photometric-distortion/color',
                'feature/photometric-distortion/noise',
              ]
            },
          ]
        },
      ]
    },
    {
      type: 'category',
      collapsible: true,
      collapsed: false,
      label: 'Utility Type',
      items: [
        'utility/image',
        'utility/label',
      ]
    }
  ]
};

module.exports = sidebars;
