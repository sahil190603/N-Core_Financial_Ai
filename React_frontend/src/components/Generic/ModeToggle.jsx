import React from 'react';
import { Switch } from 'antd';
import { SunOutlined, MoonOutlined } from '@ant-design/icons';

function ModeToggle ({ isDarkMode, onToggle })  {
    return (
        <div>
            <Switch
                checked={isDarkMode}
                onChange={onToggle}
                checkedChildren={<MoonOutlined />}
                unCheckedChildren={<SunOutlined />} 
            />
        </div>
    );
}

export default ModeToggle;
